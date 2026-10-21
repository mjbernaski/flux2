"""Build a video_server.py workflow pair from one of ComfyUI's own templates.

    python workflow_from_template.py video_minimax_h3_t2v --out h3_t2va
    python workflow_from_template.py video_minimax_h3_i2v --out h3_fl2va
    python workflow_from_template.py --list

This is what ComfyUI's "Workflow -> Export (API)" does in the browser, plus the
map file `video_server.py` needs. It exists because the browser route is not
always available (on this box Chrome cannot reach 127.0.0.1:8188 at all, while
curl gets a clean 200), and because doing it here makes regenerating after a
ComfyUI upgrade one command instead of a click-through.

Two mappings make the conversion exact rather than guesswork:

  link inputs   - a UI node carries its own `inputs` array, and
                  `inputs[target_slot]["name"]` is the input's real name.
  widget values - `widgets_values` aligns with the node class's widget inputs
                  in INPUT_TYPES declaration order, which is how the frontend
                  builds its widget list, so /object_info gives the names. A
                  widget converted to an input still occupies its slot in
                  widgets_values, so links are applied last and win.

ComfyUI's H3 templates wrap everything in a **subgraph**: the instance node's
`type` is a UUID and the real 21 nodes live under `definitions.subgraphs`. That
subgraph is inlined here. A boundary input is resolved in this order:

  1. explicitly dropped (`last_frame`, which nothing supplies)
  2. a wire type (IMAGE...) fed by an outer node -> that node is carried in and
     wired (this is how the i2v template's LoadImage arrives)
  3. a widget value from the instance's own widgets_values -> a literal
  4. otherwise dropped

Step 3 is why `width`/`height` become literals and the template's
ResolutionSelector is simply left behind: video_server.py sends an explicit
size with every request, so a node computing one is a node disagreeing with it.

The ComfyUI server must be running: /object_info is the source of truth for
widget names, and a node class the running install does not have is an error
here rather than a mystery at submit time.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HERE = os.path.abspath(os.path.dirname(__file__))
OUT_DIR = os.path.join(HERE, "workflows")
COMFY = (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")

# Input types that are wires between nodes, never literals.
LINK_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK", "AUDIO",
    "VIDEO", "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET", "STYLE_MODEL",
    "CLIP_VISION", "CLIP_VISION_OUTPUT", "GLIGEN", "UPSCALE_MODEL", "WEBCAM",
    "PHOTOMAKER", "LORA_MODEL", "MODEL_PATCH",
}

# Boundary inputs nothing supplies. last_frame is a real H3 feature (first-and-
# last-frame interpolation) but video_server.py takes one image, so wiring it
# would leave an input pointing at nothing.
DROP_BOUNDARY = ("last_frame",)


def templates_dir():
    try:
        import comfyui_workflow_templates_json as pkg
        return os.path.join(os.path.dirname(pkg.__file__), "templates")
    except ImportError:
        pass
    # The templates live in ComfyUI's venv, not this repo's, so fall back to
    # locating them under COMFY_HOME rather than requiring the import.
    home = os.environ.get("COMFY_HOME") or r"E:\ComfyUI-h3"
    for venv in ("comfy-venv", ".venv", "venv"):
        candidate = os.path.join(home, venv, "Lib", "site-packages",
                                 "comfyui_workflow_templates_json", "templates")
        if os.path.isdir(candidate):
            return candidate
    raise SystemExit("could not find ComfyUI's workflow templates - set COMFY_HOME")


def fetch_json(path):
    try:
        with urllib.request.urlopen(f"{COMFY}{path}", timeout=180) as fh:
            return json.load(fh)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"ComfyUI is not reachable at {COMFY} ({exc}). "
                         f"Start it first: switch_mode.ps1 video")


def widget_names(cls_info):
    """Widget input names of a node class, in declaration order."""
    names = []
    spec = cls_info.get("input") or {}
    for section in ("required", "optional"):
        for name, definition in (spec.get(section) or {}).items():
            if not isinstance(definition, (list, tuple)) or not definition:
                continue
            type_spec = definition[0]
            # A list of choices is a COMBO widget; a bare type name is a widget
            # only when it is not one of the wire types above.
            if isinstance(type_spec, list):
                names.append(name)
            elif isinstance(type_spec, str) and type_spec not in LINK_TYPES:
                names.append(name)
    return names


def link_parts(link):
    """(id, origin_id, origin_slot, target_id, target_slot) from either shape.

    Subgraph links are objects; the outer graph's are arrays. Both appear in
    the same file.
    """
    if isinstance(link, dict):
        return (link.get("id"), link["origin_id"], link["origin_slot"],
                link["target_id"], link["target_slot"])
    return (link[0], link[1], link[2], link[3], link[4])


def api_node(cls, node, info):
    if cls not in info:
        raise SystemExit(f"node class {cls!r} is not registered in this ComfyUI install")
    inputs = dict(zip(widget_names(info[cls]), node.get("widgets_values") or []))
    return {"class_type": cls, "inputs": inputs,
            "_meta": {"title": node.get("title") or info[cls].get("display_name") or cls}}


def convert(template_name, overrides=None):
    path = os.path.join(templates_dir(), template_name + ".json")
    if not os.path.exists(path):
        raise SystemExit(f"no template {template_name!r} in {templates_dir()}")
    with open(path, encoding="utf-8") as fh:
        ui = json.load(fh)

    info = fetch_json("/api/object_info")
    subs = (ui.get("definitions") or {}).get("subgraphs") or []
    if len(subs) != 1:
        raise SystemExit(f"expected exactly one subgraph, found {len(subs)} - this "
                         f"converter only handles the H3 template shape")
    sub = subs[0]
    in_node_id, out_node_id = sub["inputNode"]["id"], sub["outputNode"]["id"]

    outer = {n["id"]: n for n in ui["nodes"] if n.get("type") != "MarkdownNote"}
    instance = next((n for n in outer.values() if n.get("type") == sub["id"]), None)
    if instance is None:
        raise SystemExit("the subgraph is defined but never instantiated")

    # Boundary values: sub["inputs"] is in the same order as the instance's
    # widgets_values once the wire-typed inputs are set aside.
    boundary_names = [i["name"] for i in sub["inputs"]]
    wire_names = [i["name"] for i in sub["inputs"] if i["type"] in LINK_TYPES]
    widget_boundary = [n for n in boundary_names if n not in wire_names]
    values = instance.get("widgets_values") or []
    if len(widget_boundary) != len(values):
        raise SystemExit(f"boundary/widget mismatch: {len(widget_boundary)} widget inputs "
                         f"vs {len(values)} values\n{widget_boundary}\n{values}")
    boundary = dict(zip(widget_boundary, values))
    for key, value in (overrides or {}).items():
        if key not in boundary:
            raise SystemExit(f"override {key!r} is not a boundary input of this template "
                             f"({', '.join(widget_boundary)})")
        boundary[key] = value

    # Which outer node, if any, feeds each wire-typed boundary input.
    outer_links = {link_parts(l)[0]: link_parts(l) for l in (ui.get("links") or [])}
    outer_feed = {}
    for slot_index, slot in enumerate(instance.get("inputs") or []):
        name, link_id = slot.get("name"), slot.get("link")
        if name in wire_names and link_id is not None and link_id in outer_links:
            _, origin_id, origin_slot, _, _ = outer_links[link_id]
            outer_feed[name] = (origin_id, origin_slot)

    api = {}
    inner = {n["id"]: n for n in sub["nodes"]}
    for node_id, node in inner.items():
        api[str(node_id)] = api_node(node["type"], node, info)

    carried = set()

    def carry(node_id):
        """Bring an outer node (and anything feeding it) into the API graph."""
        if node_id in carried:
            return
        node = outer.get(node_id)
        if node is None:
            raise SystemExit(f"outer node {node_id} is referenced but not in the graph")
        carried.add(node_id)
        api[str(node_id)] = api_node(node["type"], node, info)
        for slot_index, slot in enumerate(node.get("inputs") or []):
            link_id = slot.get("link")
            if link_id is None or link_id not in outer_links:
                continue
            _, origin_id, origin_slot, _, _ = outer_links[link_id]
            carry(origin_id)
            api[str(node_id)]["inputs"][slot["name"]] = [str(origin_id), origin_slot]

    slot_to_boundary = dict(enumerate(boundary_names))
    for link in sub["links"]:
        _, origin_id, origin_slot, target_id, target_slot = link_parts(link)
        if target_id == out_node_id:
            continue
        target = inner.get(target_id)
        if target is None:
            raise SystemExit(f"subgraph link targets unknown node {target_id}")
        slots = target.get("inputs") or []
        if target_slot >= len(slots):
            raise SystemExit(f"link targets slot {target_slot} of node {target_id}, "
                             f"which has {len(slots)} input slots")
        input_name = slots[target_slot]["name"]

        if origin_id != in_node_id:
            api[str(target_id)]["inputs"][input_name] = [str(origin_id), origin_slot]
            continue

        name = slot_to_boundary[origin_slot]
        if name in DROP_BOUNDARY:
            api[str(target_id)]["inputs"].pop(input_name, None)
        elif name in outer_feed:
            origin, slot = outer_feed[name]
            carry(origin)
            api[str(target_id)]["inputs"][input_name] = [str(origin), slot]
        elif name in boundary:
            api[str(target_id)]["inputs"][input_name] = boundary[name]
        else:
            api[str(target_id)]["inputs"].pop(input_name, None)

    # The outer SaveVideo consumes the subgraph's VIDEO output.
    video_link = next((l for l in sub["links"] if link_parts(l)[3] == out_node_id), None)
    if video_link is None:
        raise SystemExit("the subgraph has no link into its output node")
    _, v_origin, v_slot, _, _ = link_parts(video_link)
    saver = next((n for n in outer.values() if n.get("type") == "SaveVideo"), None)
    if saver is None:
        raise SystemExit("the template has no SaveVideo node")
    api[str(saver["id"])] = api_node("SaveVideo", saver, info)
    api[str(saver["id"])]["inputs"]["video"] = [str(v_origin), v_slot]

    return api, boundary


def inline_length(api, frames):
    """Replace the seconds->frames chain with a literal frame count.

    The template feeds MiniMaxH3ImageToVideo.length from a PrimitiveFloat of
    seconds through a ComfyMathExpression running
    `max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17`. That is the
    same 17k+5 grid snap video_server.py does in _snap_frames, with a test
    asserting every value in range lands on the grid, so the chain computes
    something the request has already decided.

    It also drops the one node whose API spelling is unclear: the expression
    node's `values` input is a COMFY_AUTOGROW_V3 group whose UI slot name is
    the dotted "values.a".
    """
    target = next((nid for nid, n in api.items()
                   if n["class_type"] == "MiniMaxH3ImageToVideo"), None)
    if target is None:
        raise SystemExit("no MiniMaxH3ImageToVideo node to give a literal length")
    api[target]["inputs"]["length"] = frames
    for node_id, node in list(api.items()):
        if node["class_type"] not in ("ComfyMathExpression", "PrimitiveFloat"):
            continue
        consumers = [other for other, n in api.items()
                     if any(isinstance(v, list) and v[0] == node_id for v in n["inputs"].values())]
        if consumers:
            raise SystemExit(f"{node['class_type']} {node_id} still feeds {consumers}")
        del api[node_id]
    return api


def derive_map(api):
    """Work out the field -> "node.input" map from the graph itself.

    The node ids differ between templates and move between ComfyUI releases, so
    they are found by class and by following links. `steps` is the one that
    earns the effort: the template routes the step count through a
    ComfySwitchNode pair driven by a boolean, so the node holding the step
    count depends on whether turbo is on. Following the switch gets it right
    instead of leaving it as a footnote to be got wrong later.
    """
    def by_class(cls):
        return [nid for nid, n in api.items() if n["class_type"] == cls]

    def one(cls, what):
        found = by_class(cls)
        if len(found) != 1:
            raise SystemExit(f"expected exactly one {cls} for {what}, found {len(found)}")
        return found[0]

    h3 = one("MiniMaxH3ImageToVideo", "the prompt and size")
    field_map = {
        "prompt": f"{h3}.prompt",
        "width": f"{h3}.width",
        "height": f"{h3}.height",
        "frames": f"{h3}.length",
        "seed": f"{one('RandomNoise', 'the seed')}.noise_seed",
    }

    # steps: BasicScheduler.steps <- ComfySwitchNode <- (on_true | on_false)
    scheduler = one("BasicScheduler", "the step count")
    steps_input = api[scheduler]["inputs"].get("steps")
    if isinstance(steps_input, list):
        switch_id = steps_input[0]
        switch = api.get(switch_id, {})
        if switch.get("class_type") == "ComfySwitchNode":
            flag = switch["inputs"].get("switch")
            turbo = None
            if isinstance(flag, list):
                turbo = api.get(flag[0], {}).get("inputs", {}).get("value")
            else:
                turbo = flag
            branch = "on_true" if turbo else "on_false"
            source = switch["inputs"].get(branch)
            if isinstance(source, list) and api.get(source[0], {}).get("class_type") == "PrimitiveInt":
                field_map["steps"] = f"{source[0]}.value"
                field_map["_steps_note"] = (
                    f"turbo is {'on' if turbo else 'off'}, so the step count comes from the "
                    f"switch's {branch} branch; the other branch is wired but unused")
        elif switch.get("class_type") == "PrimitiveInt":
            field_map["steps"] = f"{switch_id}.value"
    elif steps_input is not None:
        field_map["steps"] = f"{scheduler}.steps"
    if "steps" not in field_map:
        raise SystemExit("could not work out where the step count lives")

    loaders = by_class("LoadImage")
    if len(loaders) == 1:
        field_map["start_image"] = f"{loaders[0]}.image"
    elif len(loaders) > 1:
        raise SystemExit(f"{len(loaders)} LoadImage nodes - cannot tell which is the first frame")

    # H3 has no negative prompt: MiniMaxH3ImageToVideo takes prompt, width,
    # height, length, clip, vae, first_frame and last_frame. Deliberately no
    # 'negative' entry, and video_server.py refuses a supplied one rather than
    # dropping it silently.
    return field_map


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("template", nargs="?", help="e.g. video_minimax_h3_t2v")
    parser.add_argument("--out", help="workflow name to write under workflows/ (default: the template name)")
    parser.add_argument("--list", action="store_true", help="list the H3 templates available")
    parser.add_argument("--unet", default="minimax_h3_fl2va_pruned_w6a8.safetensors",
                        help="transformer checkpoint (default: the one that fits a 32GB card)")
    parser.add_argument("--lora", default="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")
    parser.add_argument("--turbo", default="on", choices=("on", "off"))
    parser.add_argument("--turbo-steps", type=int, default=4)
    parser.add_argument("--size", default="864x480")
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--no-map", action="store_true", help="write only the graph, not the map")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.list:
        names = sorted(n[:-5] for n in os.listdir(templates_dir())
                       if n.endswith(".json") and "h3" in n.lower() and not n.startswith("api_"))
        print(f"local-weights H3 templates in {templates_dir()}:")
        for name in names:
            print("  " + name)
        print("\n(api_* templates are the paid MiniMax cloud nodes, not local weights)")
        return 0

    if not args.template:
        parser.error("a template name is required (or --list)")

    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        parser.error("--size must look like 864x480")

    overrides = {
        "unet_name": args.unet,
        "lora_name": args.lora,
        "value": args.turbo == "on",
        "value_2": args.turbo_steps,
        "width": width,
        "height": height,
    }
    api, boundary = convert(args.template, overrides=overrides)
    api = inline_length(api, args.frames)

    name = args.out or args.template
    print(f"{args.template} -> {len(api)} nodes")
    for key in ("unet_name", "clip_name", "vae_name", "vae_name_1", "lora_name", "value", "value_2"):
        if key in boundary:
            print(f"  {key:<18} = {boundary[key]!r}")
    print(f"  {'size':<18} = {width}x{height}, {args.frames} frames")

    field_map = None if args.no_map else derive_map(api)
    if field_map:
        print("\nderived map:")
        for key, value in field_map.items():
            print(f"  {key:<14} {value}")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    graph_path = os.path.join(OUT_DIR, name + ".json")
    with open(graph_path, "w", encoding="utf-8") as fh:
        json.dump(api, fh, indent=1)
    print(f"\nwrote {os.path.relpath(graph_path, HERE)}")
    if field_map:
        map_path = os.path.join(OUT_DIR, name + ".map.json")
        with open(map_path, "w", encoding="utf-8") as fh:
            json.dump(field_map, fh, indent=2)
        print(f"wrote {os.path.relpath(map_path, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
