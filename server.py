"""Standard-library HTTP adapter. No model is loaded until main()."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__:
    from .core import UnsupportedRequest, Service, ROOT, load_config, load_json, local_output, validate_config, apply_profile, validate_calibration, validate_threshold
else:
    from core import UnsupportedRequest, Service, ROOT, load_config, load_json, local_output, validate_config, apply_profile, validate_calibration, validate_threshold


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'

        def handle_expect_100(self):
            try:
                n=int(self.headers.get('Content-Length','0'))
                if n<=0 or n>service.config['server']['max_request_bytes'] or self.headers.get('Transfer-Encoding'):
                    raise ValueError
            except (ValueError,TypeError):
                self.close_connection=True
                self.respond(400,{'error':'Invalid request size'})
                return False
            self.send_response_only(100)
            self.end_headers()
            return True

        def setup(self):
            super().setup()
            self.connection.settimeout(service.config["server"]["socket_timeout_seconds"])

        def log_message(self, format, *args):
            # Avoid logging request contents, model paths or network endpoints.
            pass

        def respond(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split('?', 1)[0].rstrip('/') == '/health':
                self.respond(200, service.health())
            elif self.path.split('?',1)[0].rstrip('/') == '/engine_stats':
                try: self.respond(200,service.engine_stats())
                except Exception: self.respond(503,{'error':'Engine counters unavailable'})
            else:
                self.respond(404, {'error':'Unknown route'})

        def do_POST(self):
            if self.path.split("?", 1)[0].rstrip("/") != "/v1/systemone":
                self.close_connection=True
                self.respond(404, {"error": "Unknown route"})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is unsupported")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > service.config["server"]["max_request_bytes"]:
                    raise ValueError("Invalid request size")
                def reject(value):
                    raise ValueError("Non-finite JSON value")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                payload = json.loads(raw.decode("utf-8"), parse_constant=reject)
                # Validate before inference so model ValueError is never a client error.
                if __package__:
                    from .core import normalize_request
                else:
                    from core import normalize_request
                normalize_request(payload, service.config["server"]["max_questions"])
            except UnsupportedRequest:
                self.close_connection=True
                self.respond(422,{"error":"Unsupported systemone request"})
                return
            except (ValueError, TypeError, UnicodeError, OSError):
                self.close_connection=True
                self.respond(400, {"error": "Invalid systemone request"})
                return
            try:
                result = service.answer(payload)
                json.dumps(result,allow_nan=False)
            except Exception:
                if __package__: from .core import emergency_response
                else: from core import emergency_response
                result=emergency_response(payload,service.config)
            self.respond(200, result)

    return Handler


class QuietHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # The base implementation prints a peer endpoint on disconnected clients.
        print("HTTP connection failed", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--model")
    parser.add_argument("--quant", choices=("int8", "bf16", "awq"))
    parser.add_argument("--adapter")
    parser.add_argument("--backend", choices=("transformers", "vllm"))
    parser.add_argument("--bind", help="Required unless server.bind is set in config")
    parser.add_argument("--port", type=int)
    parser.add_argument("--mode", choices=("routed", "uniform", "fast_only"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model-key')
    parser.add_argument('--quota-tier',choices=('normal','hard','long'))
    parser.add_argument('--cost-fuse',choices=('on','off'))
    parser.add_argument('--no-prefix-caching',action='store_true')
    parser.add_argument('--allow-placeholder-threshold',action='store_true')
    args = parser.parse_args(argv)
    c = load_json(args.config)
    for arg, field in (("model", "path"), ("quant", "quant"), ("adapter", "adapter"), ("backend", "backend")):
        if getattr(args, arg) is not None:
            c["model"][field] = getattr(args, arg)
    for key in ("bind", "port"):
        if getattr(args, key) is not None:
            c["server"][key] = getattr(args, key)
    if args.mode is not None:
        c["routing"]["mode"] = args.mode
    if args.state_dir is not None:
        c['routing']['state_dir'] = str(args.state_dir)
    if args.run_id:
        c['routing']['run_id'] = args.run_id
    c['routing']['resume'] = args.resume
    if args.model_key:
        c['model']['model_key'] = args.model_key
    apply_profile(c)
    if __package__: from .core import apply_budget_tier
    else: from core import apply_budget_tier
    apply_budget_tier(c,args.quota_tier)
    if args.mode is not None:c["routing"]["mode"]=args.mode
    if args.quant is not None and args.quant!=c['model']['quant']:parser.error('Precision conflicts with model profile')
    if args.backend is not None and args.backend!=c['model']['backend']:parser.error('Backend conflicts with model profile')
    if args.no_prefix_caching: c["model"]["enable_prefix_caching"]=False
    if args.cost_fuse:c['routing']['cost_fuse']['enabled']=args.cost_fuse=='on'
    validate_config(c)
    validate_threshold(c,args.allow_placeholder_threshold)
    if not c["model"]["path"] or not c["server"]["bind"]:
        parser.error("Set --model and --bind, or their config fields")
    calibration = load_json(args.calibration or ROOT / c["calibration_file"])
    validate_calibration(calibration, c["model"]["model_key"])
    if __package__:
        from .backends import create_backend
        from .quota import RuntimeCounter
    else:
        from backends import create_backend
        from quota import RuntimeCounter
    counter = RuntimeCounter(c)
    service = Service(c, calibration, create_backend(c), counter, allow_placeholder_threshold=args.allow_placeholder_threshold)
    if c["server"].get("warmup", True):
        service.warmup()
    httpd = QuietHTTPServer((c["server"]["bind"], c["server"]["port"]), make_handler(service))
    print("systemone service ready", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
