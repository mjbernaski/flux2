//! CLI demos for the FLUX REST API — one subcommand per major capability.
//!
//! ```text
//! export FLUX_API_KEY=your_key
//! cargo run -- health
//! cargo run -- generate "a red fox in falling snow" --steps 30
//! cargo run -- batch "a lighthouse at dusk" --count 4
//! cargo run -- img2img photo.jpg "make it winter"
//! cargo run -- describe photo.jpg
//! cargo run -- boost "a castle" --level 4
//! cargo run -- history
//! cargo run -- models
//! cargo run -- multirun "a red fox" 9 6 1
//! ```

use std::io::Write;

use flux_client::{encode_image, Error, FluxClient, GenerateRequest, Job};

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        match &e {
            Error::Api { code, status, .. } => {
                eprintln!("\nServer rejected the request — code={code} (HTTP {status})");
                eprintln!("  {e}");
            }
            Error::Transport(_) => {
                eprintln!("\nCould not reach the server. Is it running? ({e})");
            }
            _ => eprintln!("\n{e}"),
        }
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Error> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");
    let flux = FluxClient::from_env()?;

    match command {
        "health" => health(&flux).await,
        "generate" => generate(&flux, &args).await,
        "batch" => batch(&flux, &args).await,
        "img2img" => img2img(&flux, &args).await,
        "inpaint" => inpaint(&flux, &args).await,
        "describe" => describe(&flux, &args).await,
        "boost" => boost(&flux, &args).await,
        "queue" => queue(&flux).await,
        "history" => history(&flux).await,
        "models" => models(&flux).await,
        "switch" => switch(&flux, &args).await,
        "multirun" => multirun(&flux, &args).await,
        "browse" => browse(&flux, &args).await,
        "errors" => errors(&flux).await,
        _ => {
            println!("{}", HELP);
            Ok(())
        }
    }
}

const HELP: &str = "\
flux — sample client for the FLUX image generator REST API

  health                       readiness and capabilities
  generate <prompt> [--steps N] [--seed N] [--size 1mp]
  batch <prompt> [--count N]   several images from one prompt
  img2img <image> <prompt> [--strength 0.5]
  inpaint <image> <mask> <prompt>
  describe <image>             photo -> prompt, via the vision model
  boost <prompt> [--level 3]   rewrite a draft prompt
  queue                        what is running and what is waiting
  history                      today's images
  models                       list configs
  switch <id>                  switch config (restarts the server)
  multirun <prompt> <id>...    one prompt across several configs
  browse [dir]                 server-side folders
  errors                       how failures surface

Set FLUX_API_KEY, and FLUX_URL for a remote host.";

/// Flag parsing kept deliberately tiny so the API calls stay the focus.
fn flag<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    let pos = args.iter().position(|a| a == name)?;
    args.get(pos + 1)?.parse().ok()
}

fn positional(args: &[String], index: usize) -> Option<&String> {
    args.iter().filter(|a| !a.starts_with("--")).nth(index)
}

/// A one-line progress bar for wait_for_job's callback.
fn progress(job: &Job) {
    if job.state == "running" && job.total_steps > 0 {
        let width = 24usize;
        let filled = (width * job.step as usize) / job.total_steps.max(1) as usize;
        let bar: String = "#".repeat(filled) + &".".repeat(width - filled);
        print!(
            "\r  image {}/{} [{}] step {}/{}",
            job.current.max(1),
            job.batch.max(1),
            bar,
            job.step,
            job.total_steps
        );
        let _ = std::io::stdout().flush();
    } else if job.is_finished() {
        println!("\r  {}{}", job.state, " ".repeat(44));
    }
}

async fn health(flux: &FluxClient) -> Result<(), Error> {
    let state = flux.health().await?;
    println!(
        "ready:  {}  ({}, {:.1}s)",
        state.ready, state.status, state.elapsed_s
    );
    if !state.ready {
        println!("The model is still loading; generation calls will 503 until it is up.");
        return Ok(());
    }

    let info = flux.model().await?;
    println!("model:  {}", info.description);
    let caps: Vec<&str> = [
        ("negative prompts", info.negative_prompt),
        ("inpainting", info.inpaint),
        ("kontext editing", info.kontext),
        ("turbo LoRA", info.turbo),
    ]
    .iter()
    .filter(|(_, on)| *on)
    .map(|(name, _)| *name)
    .collect();
    println!(
        "can:    {}",
        if caps.is_empty() {
            "text-to-image only".to_string()
        } else {
            caps.join(", ")
        }
    );

    let queue = flux.jobs().await?;
    println!(
        "queue:  {} waiting, {}",
        queue.queued.len(),
        if queue.running.is_some() {
            "1 running"
        } else {
            "idle"
        }
    );
    Ok(())
}

async fn generate(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let prompt = positional(args, 1).cloned().unwrap_or_else(|| {
        eprintln!("usage: generate <prompt>");
        std::process::exit(2);
    });

    flux.wait_until_ready(Some(&|s| println!("  {s}...."))).await?;
    println!("Generating: {prompt:?}");

    let mut request = GenerateRequest::new(&prompt).steps(flag(args, "--steps").unwrap_or(25));
    if let Some(seed) = flag::<i64>(args, "--seed") {
        request = request.seed(seed);
    }
    if let Some(size) = flag::<String>(args, "--size") {
        request = request.size(size);
    }

    let job = flux.generate(&request, Some(&progress)).await?;
    for image in &job.images {
        let path = flux.download(&image.filename, ".").await?;
        println!("  saved {}  (seed {})", path.display(), image.seed);
    }
    println!("  {:.1}s total", job.generation_time);
    Ok(())
}

/// The server encodes the prompt once for a whole batch, so this is markedly
/// cheaper than N separate jobs.
async fn batch(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let prompt = positional(args, 1).cloned().unwrap_or_default();
    let count: u32 = flag(args, "--count").unwrap_or(4);
    flux.wait_until_ready(None).await?;
    println!("Generating {count} variations of {prompt:?}");

    let request = GenerateRequest::new(&prompt)
        .batch(count)
        .steps(flag(args, "--steps").unwrap_or(25));
    let job = flux.generate(&request, Some(&progress)).await?;
    for image in &job.images {
        println!(
            "  {}  (seed {})",
            flux.download(&image.filename, ".").await?.display(),
            image.seed
        );
    }
    Ok(())
}

async fn img2img(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let image = positional(args, 1).cloned().unwrap_or_default();
    let prompt = positional(args, 2).cloned().unwrap_or_default();
    flux.wait_until_ready(None).await?;

    println!("Editing {image} with {}", flux.model().await?.model);
    let request = GenerateRequest::new(&prompt)
        .reference(encode_image(&image)?)
        .strength(flag(args, "--strength").unwrap_or(0.5))
        .keep_aspect();

    let job = flux.generate(&request, Some(&progress)).await?;
    for out in &job.images {
        println!("  {}", flux.download(&out.filename, ".").await?.display());
    }
    Ok(())
}

/// Masked editing: white in the mask marks the region to regenerate. Needs a
/// FLUX.2 or SDXL backend and exactly one reference.
async fn inpaint(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let image = positional(args, 1).cloned().unwrap_or_default();
    let mask = positional(args, 2).cloned().unwrap_or_default();
    let prompt = positional(args, 3).cloned().unwrap_or_default();

    flux.wait_until_ready(None).await?;
    if !flux.model().await?.inpaint {
        println!("This backend cannot inpaint — start the server with a FLUX.2 config.");
        return Ok(());
    }

    let request = GenerateRequest::new(&prompt)
        .reference(encode_image(&image)?)
        .mask(encode_image(&mask)?);
    let job = flux.generate(&request, Some(&progress)).await?;
    for out in &job.images {
        println!("  {}", flux.download(&out.filename, ".").await?.display());
    }
    Ok(())
}

async fn describe(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let image = positional(args, 1).cloned().unwrap_or_default();
    println!("Describing {image}...");
    let prompt = flux.describe(&[encode_image(&image)?], false).await?;
    println!("\n{prompt}\n");
    Ok(())
}

async fn boost(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let prompt = positional(args, 1).cloned().unwrap_or_default();
    let level: u32 = flag(args, "--level").unwrap_or(3);
    println!("Boosting (level {level}): {prompt:?}");
    println!("\n{}\n", flux.boost(&prompt, level, false).await?);
    Ok(())
}

async fn queue(flux: &FluxClient) -> Result<(), Error> {
    let q = flux.queue().await?;

    match &q.running {
        Some(job) => {
            println!(
                "running   {}  image {}/{} step {}/{}",
                job.id,
                job.current.max(1),
                job.batch.max(1),
                job.step,
                job.total_steps
            );
            println!("          {}", job.prompt.chars().take(70).collect::<String>());
        }
        None => println!("running   (idle)"),
    }

    println!(
        "\nwaiting   {}/{}{}",
        q.depth,
        q.capacity,
        if q.accepting {
            ""
        } else {
            "  — FULL, new jobs are rejected"
        }
    );
    for job in &q.waiting {
        println!(
            "  {:>2}.  {}  x{}  {}",
            job.position,
            job.id,
            job.batch,
            job.prompt.chars().take(56).collect::<String>()
        );
    }

    match (q.estimated_wait_s, q.seconds_per_image) {
        (Some(wait), Some(per)) => println!(
            "\n{} image(s) pending, about {wait:.0}s at {per:.1}s each",
            q.images_pending
        ),
        _ if q.images_pending > 0 => println!(
            "\n{} image(s) pending (no completed job yet to estimate from)",
            q.images_pending
        ),
        _ => {}
    }
    Ok(())
}

async fn history(flux: &FluxClient) -> Result<(), Error> {
    let images = flux.history().await?;
    println!("{} image(s) generated today", images.len());
    for image in images.iter().take(20) {
        let prompt = image.prompt.clone().unwrap_or_default();
        println!(
            "  {}  {}  {}",
            image.time,
            image.filename,
            prompt.chars().take(64).collect::<String>()
        );
    }
    Ok(())
}

async fn models(flux: &FluxClient) -> Result<(), Error> {
    let list = flux.models().await?;
    for config in &list.configs {
        let marker = if Some(config.id) == list.current {
            " <- running"
        } else {
            ""
        };
        println!("  {:>2}  {}{}", config.id, config.label, marker);
    }
    if !list.switchable {
        println!("\nNot switchable: started without the supervisor, so it cannot restart itself.");
    }
    Ok(())
}

async fn switch(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let config: u32 = positional(args, 1).and_then(|s| s.parse().ok()).unwrap_or(9);
    println!("Switching to config {config}...");
    flux.switch_model(config).await?;

    // The old process needs a moment to exit, or the first health check would
    // be answered by the server that is about to die.
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    flux.wait_until_ready(Some(&|s| println!("  {s}..."))).await?;
    println!("Now running: {}", flux.model().await?.description);
    Ok(())
}

async fn multirun(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let prompt = positional(args, 1).cloned().unwrap_or_default();
    let configs: Vec<u32> = args
        .iter()
        .skip(2)
        .filter_map(|a| a.parse().ok())
        .collect();

    let run = flux.multi_run(&prompt, &configs, Some(25)).await?;
    println!(
        "Run {} over configs {:?} (shared seed {})",
        run.id, run.configs, run.seed
    );
    println!("The server restarts between models; this follows it through.\n");

    let mut seen = 0usize;
    loop {
        match flux.multi_run_status().await {
            Ok(None) => break,
            Ok(Some(state)) => {
                if let Some(results) = state.run["results"].as_array() {
                    for result in results.iter().skip(seen) {
                        println!(
                            "  {}: {}",
                            result["label"].as_str().unwrap_or("?"),
                            result["state"].as_str().unwrap_or("?")
                        );
                    }
                    seen = results.len();
                }
                if !state.active {
                    break;
                }
            }
            // Expected while the server is mid-restart between configs.
            Err(Error::Transport(_)) => {}
            Err(e) => return Err(e),
        }
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
    println!("\nDone");
    Ok(())
}

async fn browse(flux: &FluxClient, args: &[String]) -> Result<(), Error> {
    let listing = flux.browse(positional(args, 1).map(String::as_str)).await?;
    println!("{}", listing.dir);
    for name in listing.dirs.iter().take(20) {
        println!("  [dir]  {name}");
    }
    for file in listing.files.iter().take(20) {
        println!("         {}", file.filename);
    }
    println!(
        "\n{} folder(s), {} image(s). Use these paths with GenerateRequest::server_path \
         to skip uploading.",
        listing.dirs.len(),
        listing.files.len()
    );
    Ok(())
}

/// Every failure carries a stable `code`; match on that, not the message.
async fn errors(flux: &FluxClient) -> Result<(), Error> {
    println!("Errors are {{\"error\": {{\"code\", \"message\"}}}} — match on code:\n");

    if let Err(e) = flux.job("nosuchjob1234").await {
        println!("  unknown job      code={:?}", e.code());
    }
    if let Err(e) = flux.submit(&GenerateRequest::new("")).await {
        println!("  empty prompt     code={:?}", e.code());
    }
    if let Err(e) = flux
        .submit(&GenerateRequest::new("x").steps(9999))
        .await
    {
        println!("  bad step count   code={:?}", e.code());
    }
    if let Err(e) = flux.import_url("ftp://example.com/x.png").await {
        println!("  bad import url   code={:?}", e.code());
    }
    Ok(())
}
