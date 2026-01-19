import os
import sys
from pathlib import Path
from io import BytesIO
from typing import Optional

# 親ディレクトリをsys.pathに追加（coreモジュールをインポートするため）
# __file__が存在する場合（スクリプトとして実行）は親ディレクトリを追加
if "__file__" in globals():
    parent_dir = Path(__file__).parent.parent.resolve()
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import StreamingConfig
from core.processor import StreamingAudioProcessor
from core.session_store import SessionStore

app = FastAPI(title="Realtime ASR FasterWhisper Service", version="0.1.0")

config = StreamingConfig()
processor = StreamingAudioProcessor(config)
session_store = SessionStore(ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")))


class ResetRequest(BaseModel):
    session_id: str


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


@app.get("/")
async def root():
    """ルートエンドポイント - API情報を返す"""
    return {
        "service": "Realtime ASR FasterWhisper Service",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "stream": "/stream",
            "vad": "/vad",
            "transcribe": "/transcribe",
            "session_reset": "/session/reset",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/stream")
async def stream_audio(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    end_of_stream: Optional[str] = Form(None),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio, samplerate = sf.read(BytesIO(data))
        audio = np.asarray(audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}")

    session_id, state = session_store.get_or_create(session_id)
    try:
        _, text, is_final = processor.process_chunk(state, audio, int(samplerate))
        if _parse_bool(end_of_stream):
            _, text, is_final = processor.flush(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Streaming processing failed: {exc}")

    return {"session_id": session_id, "text": text or "", "is_final": is_final}


@app.post("/vad")
async def vad(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio, samplerate = sf.read(BytesIO(data))
        audio = np.asarray(audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}")

    audio_16k = processor.preprocess_audio(audio, int(samplerate))
    segments = processor.detect_speech(audio_16k)
    segments_out = [
        {
            "start": int(seg["start"]),
            "end": int(seg["end"]),
            "start_sec": float(seg["start"]) / 16000,
            "end_sec": float(seg["end"]) / 16000,
        }
        for seg in segments
    ]
    return {
        "has_speech": len(segments_out) > 0,
        "segments": segments_out,
        "sample_rate": 16000,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    use_vad: Optional[str] = Form(None),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        audio, samplerate = sf.read(BytesIO(data))
        audio = np.asarray(audio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {exc}")

    audio_16k = processor.preprocess_audio(audio, int(samplerate))
    use_vad_flag = _parse_bool(use_vad)
    segments_out = []
    if use_vad_flag:
        segments = processor.detect_speech(audio_16k)
        segments_out = [
            {
                "start": int(seg["start"]),
                "end": int(seg["end"]),
                "start_sec": float(seg["start"]) / 16000,
                "end_sec": float(seg["end"]) / 16000,
            }
            for seg in segments
        ]
        if segments:
            audio_16k = np.concatenate([audio_16k[seg["start"] : seg["end"]] for seg in segments])
        else:
            audio_16k = np.zeros(0, dtype=np.float32)

    text = processor.transcribe_audio(audio_16k) if len(audio_16k) > 0 else ""
    return {"text": text or "", "use_vad": use_vad_flag, "segments": segments_out}


@app.post("/session/reset")
async def reset_session(req: ResetRequest):
    if not session_store.reset(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    import subprocess
    import threading
    import time

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9100"))
    use_share = os.getenv("USE_SHARE", "").lower() in {"1", "true", "yes", "on"}
    share_type = os.getenv("SHARE_TYPE", "cloudflared").lower()  # "cloudflared" or "ngrok"
    
    def start_tunnel():
        """トンネルを開始して公開URLを取得"""
        time.sleep(2)  # サーバーが起動するまで待機
        
        if share_type == "cloudflared":
            try:
                # cloudflaredを使用（無料、アカウント不要）
                print("[share] Cloudflared トンネルを開始しています...", flush=True)
                import re
                
                process = subprocess.Popen(
                    ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # stderrをstdoutにリダイレクト
                    text=True,
                    bufsize=1  # 行バッファリング
                )
                
                # cloudflaredの出力からURLを取得（正規表現で抽出）
                url_pattern = re.compile(r'https://[a-z0-9-]+\.trycloudflare\.com')
                public_url = None
                
                for line in process.stdout:
                    print(f"[cloudflared] {line}", end='', flush=True)  # デバッグ用に出力
                    # URLパターンを探す
                    match = url_pattern.search(line)
                    if match:
                        public_url = match.group(0)
                        print(f"\n[share] 公開URL: {public_url}", flush=True)
                        print(f"[share] ローカルURL: http://{host}:{port}", flush=True)
                        break
                
                if not public_url:
                    print("[share] 警告: 公開URLを取得できませんでした", flush=True)
                    
            except FileNotFoundError:
                print("[share] cloudflaredが見つかりません。インストールしてください:", flush=True)
                print("  macOS: brew install cloudflared", flush=True)
                print("  Linux: https://github.com/cloudflare/cloudflared/releases", flush=True)
                print("  Docker: コンテナにcloudflaredをインストールしてください", flush=True)
            except Exception as e:
                print(f"[share] トンネルの開始に失敗しました: {e}", flush=True)
        
        elif share_type == "ngrok":
            try:
                # pyngrokを使用
                from pyngrok import ngrok
                ngrok_token = os.getenv("NGROK_AUTHTOKEN")
                if ngrok_token:
                    ngrok.set_auth_token(ngrok_token)
                
                print("[share] Ngrok トンネルを開始しています...", flush=True)
                public_url = ngrok.connect(port, "http")
                print(f"[share] 公開URL: {public_url}", flush=True)
                print(f"[share] ローカルURL: http://{host}:{port}", flush=True)
                
                # ngrokのダッシュボードURLも表示
                tunnel = ngrok.get_tunnels()[0]
                if tunnel.public_url:
                    print(f"[share] Ngrok ダッシュボード: http://127.0.0.1:4040", flush=True)
            except ImportError:
                print("[share] pyngrokがインストールされていません。インストールしてください:", flush=True)
                print("  pip install pyngrok", flush=True)
            except Exception as e:
                print(f"[share] トンネルの開始に失敗しました: {e}", flush=True)
    
    if use_share:
        # バックグラウンドでトンネルを開始
        tunnel_thread = threading.Thread(target=start_tunnel, daemon=True)
        tunnel_thread.start()
    
    print(f"[server] サーバーを開始しています: http://{host}:{port}", flush=True)
    if use_share:
        print(f"[server] 公開リンク機能: 有効 ({share_type})", flush=True)
    
    uvicorn.run(app, host=host, port=port)
