#!/usr/bin/env python3
"""Compact web UI for the FLUX image generator, on port 3333.

A second front end over the same generation server, deliberately kept out of
`web_server.py`: the generator restarts on every model switch (the exit-86
flow), and a UI that dies with it cannot show you the restart it triggered.
Running here means 3333 stays up across a switch and can narrate it.

    python new_ui.py [--port 3333] [--upstream http://127.0.0.1:2222]

It talks to nothing but `/api/v1` — the documented REST dialect, not the
legacy flat routes — so this file is also the largest working example of that
API. Everything it needs is proxied:

    browser  ->  :3333/api/v1/*  ->  :2222/api/v1/*

The proxy is the reason there is a server here at all rather than a folder of
static files. It keeps the browser same-origin (no CORS on the generator, no
preflight on every poll) and lets `FLUX_API_KEY` live in this process instead
of in the page — a request that arrives without a key gets the server's. A
browser that does send `X-API-Key` wins, so the key box in the UI still works
for a remote user who has one.
"""

import argparse
import os

import requests
from dotenv import load_dotenv
from flask import Flask, Response, request, send_from_directory

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT, 'static-v2')

UPSTREAM = os.environ.get('FLUX_UPSTREAM', 'http://127.0.0.1:2222')
API_KEY = os.environ.get('FLUX_API_KEY', '')

# Long enough for a cold model load to answer /health, short enough that a
# genuinely dead upstream fails the poll instead of piling up sockets.
UPSTREAM_TIMEOUT = 300

# Hop-by-hop headers must not be forwarded; Flask sets its own.
STRIPPED = {'content-encoding', 'content-length', 'transfer-encoding',
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailer', 'upgrade'}

app = Flask(__name__, static_folder=None)


@app.after_request
def _no_store(response):
    """The UI is polled hard and edited often; a cached app.js or a cached
    /queue is worse than a re-fetch on every load."""
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:filename>')
def static_file(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route('/api/v1', defaults={'sub': ''},
           methods=['GET', 'POST', 'PUT', 'DELETE'])
@app.route('/api/v1/<path:sub>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(sub):
    """Forward one call to the generation server and hand back its answer.

    The body is passed through untouched — `/imports/raw` is multipart and
    every other POST is JSON, and neither needs parsing here to be relayed.
    """
    target = f"{UPSTREAM}/api/v1/{sub}" if sub else f"{UPSTREAM}/api/v1"

    headers = {'X-API-Key': request.headers.get('X-API-Key') or API_KEY}
    if request.content_type:
        headers['Content-Type'] = request.content_type

    try:
        upstream = requests.request(
            request.method, target,
            params=request.args,
            data=request.get_data(),
            headers=headers,
            timeout=UPSTREAM_TIMEOUT,
        )
    except requests.RequestException as e:
        # The generator is legitimately absent during a model switch, so this
        # is a state the UI renders rather than an error it reports.
        return {'error': {'code': 'upstream_unreachable', 'message': str(e)}}, 503

    passthrough = [(k, v) for k, v in upstream.headers.items()
                   if k.lower() not in STRIPPED]
    return Response(upstream.content, status=upstream.status_code,
                    headers=passthrough)


def main():
    global UPSTREAM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=3333)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--upstream', default=UPSTREAM,
                        help='Base URL of the generation server (port 2222)')
    args = parser.parse_args()

    UPSTREAM = args.upstream.rstrip('/')

    print(f"Compact UI on http://localhost:{args.port}")
    print(f"  proxying /api/v1 -> {UPSTREAM}/api/v1")
    print(f"  server-side API key: {'set' if API_KEY else 'NOT SET (browser must supply one)'}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
