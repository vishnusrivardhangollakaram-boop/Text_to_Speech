"""
Image Storyteller — 上传图片 → 生成英文故事 → 朗读

技术栈：
  - 读图提取细节：Hugging Face Transformers pipeline("image-to-text")
                    (Salesforce/blip-image-captioning-base)
  - 故事生成：     pipeline("text-generation") (distilgpt2)
  - 可选 LLM 方式：HF Inference API 远程调用 VLM
  - 朗读：         edge-tts（微软 Edge 在线 TTS）
  - UI：           Streamlit（暖纸质感主题，全英文，全直角）

部署：Streamlit Community Cloud
"""

import asyncio
import base64
import gc
import io

import edge_tts
import streamlit as st
from PIL import Image, ImageOps
from transformers import pipeline

# ============================================================================ #
# 配置常量
# ============================================================================ #
CAPTION_MODEL = "microsoft/Florence-2-base"   # 读图（固定，出详细 caption）
HF_BASE_URL = "https://api-inference.huggingface.co/v1"    # LLM 远程端点

MAX_IMAGE_SIZE = 1024        # 推理前图片长边上限
PREVIEW_SIZE = 800           # 相框预览图长边上限
MAX_UPLOAD_BYTES = 12 * 1024 * 1024   # 12MB 上传上限
ALLOWED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

# 编故事模型（generation mode 只切换「编故事」模型，读图固定 Florence-2）
STORY_MODELS = {
    "StorySupra-10M (local pipeline)": {"kind": "pipeline", "id": "SupraLabs/StorySupra-10M"},
    "distilgpt2 (local pipeline)": {"kind": "pipeline", "id": "distilgpt2"},
    "GLM-OCR (LLM API)": {"kind": "llm", "id": "zai-org/GLM-OCR"},
    "Qwen2.5-VL-7B (LLM API)": {"kind": "llm", "id": "Qwen/Qwen2.5-VL-7B-Instruct"},
}

# 风格 / 长度 / 音色 / 语速
STORY_STYLES = {
    "Fairy tale": "a whimsical fairy tale",
    "Sci-fi": "a short science-fiction story",
    "Mystery": "a suspenseful mystery story",
    "Heartwarming": "a warm, heartwarming story",
    "Adventure": "a thrilling adventure story",
    "Poem": "a short poem",
}
LENGTHS = {
    "Short (~50 words)": 50,
    "Medium (~75 words)": 75,
    "Long (~100 words)": 100,
}
VOICES = {
    "Aria · US female": "en-US-AriaNeural",
    "Jenny · US female": "en-US-JennyNeural",
    "Emma · US female": "en-US-EmmaNeural",
    "Guy · US male": "en-US-GuyNeural",
    "Andrew · US male": "en-US-AndrewNeural",
    "Sonia · UK female": "en-GB-SoniaNeural",
    "Ryan · UK male": "en-GB-RyanNeural",
}
SPEEDS = {"Slow": "-20%", "Normal": "+0%", "Fast": "+20%"}


# ============================================================================ #
# 页面配置
# ============================================================================ #
st.set_page_config(page_title="Image Storyteller", page_icon="🖼️", layout="centered")


# ============================================================================ #
# 图片工具
# ============================================================================ #
def validate_size(uploaded) -> str | None:
    """校验文件大小（格式已由上传白名单限制）。返回错误信息，合法则返回 None。"""
    if uploaded.size > MAX_UPLOAD_BYTES:
        return "File is too large. Please upload an image under 12 MB."
    return None


def load_image(uploaded) -> Image.Image:
    """读取图片并统一为 RGB，同时应用 EXIF 旋转（避免手机照片横躺）。"""
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img)   # 应用方向信息
    return img.convert("RGB")


def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """按长边等比缩放，避免超大图占用内存 / 超出模型输入。"""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def image_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """PIL 图片 → base64 字符串（不含 data: 前缀）。"""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================================ #
# 模型推理（pipeline 主路径）
# ============================================================================ #
def extract_details(image: Image.Image) -> str:
    """
    读图提取细节（image-to-text pipeline）。
    说明：此处按需加载模型、用完即释放（函数返回后局部变量被 GC），
    以避免 Cloud 1GB 内存同时容纳两个模型而 OOM。
    """
    captioner = pipeline("image-text-to-text", model=CAPTION_MODEL)
    try:
        result = captioner(image, text="<MORE_DETAILED_CAPTION>")[0]["generated_text"]
        return result.strip()
    finally:
        del captioner
        gc.collect()


def build_story_prompt(details: str, style_desc: str, words: int) -> str:
    """构造故事生成提示词。"""
    return (
        f"Here is a scene: {details}. "
        f"Write {style_desc} based on it, about {words} words.\n\n"
    )


def trim_to_sentence_boundary(text: str, max_words: int) -> str:
    """把文本裁剪到目标词数附近的完整句子边界，避免半句话收尾。"""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_punct = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_punct > 0:
        truncated = truncated[: last_punct + 1]
    return truncated


def generate_story_pipeline(details: str, style_desc: str, words: int, model_id: str) -> str:
    """
    基于图片细节生成英文故事（text-generation pipeline）。
    同样按需加载、用完释放。
    """
    prompt = build_story_prompt(details, style_desc, words)
    generator = pipeline("text-generation", model=model_id)
    try:
        out = generator(
            prompt,
            max_new_tokens=words * 3,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        full = out[0]["generated_text"]
        return full[len(prompt):].strip()
    finally:
        del generator
        gc.collect()


# ============================================================================ #
# 模型推理（LLM 远程 API 选项）
# ============================================================================ #
def generate_story_llm(
    details: str, model_id: str, style_desc: str, words: int, token: str
) -> str:
    """通过 HF Inference API（OpenAI 兼容）远程调用 LLM，基于图片细节编故事。"""
    from openai import OpenAI  # 延迟导入，仅在 LLM 模式需要

    client = OpenAI(base_url=HF_BASE_URL, api_key=token)
    prompt = (
        f"Here is a detailed description of a scene: {details}. "
        f"Write {style_desc} based on it, around {words} words. "
        f"Write only the story itself in English."
    )
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.85,
    )
    return resp.choices[0].message.content.strip()


def friendly_hf_error(e: Exception) -> str:
    """把 HF 常见错误翻译成对用户友好的英文提示。"""
    msg = str(e)
    if "401" in msg:
        return "Invalid or unauthorized HF Token. Please check the token in the sidebar."
    if "403" in msg or "gated" in msg.lower() or "Authorization" in msg:
        return "This model is gated. Accept its license on Hugging Face, or your token lacks access."
    if "429" in msg:
        return "Rate limited: free quota exhausted, or this model requires Inference PRO."
    if "404" in msg or "not found" in msg.lower():
        return "Model not found. Please check the model ID."
    if "too large" in msg.lower():
        return "Model is too large for the free tier. It requires Inference PRO."
    return f"Generation failed: {msg}"


def generate_story(
    image: Image.Image,
    story_model: dict,
    style_desc: str,
    words: int,
    token: str,
) -> str:
    """故事生成编排器：读图固定 Florence-2，编故事按所选模型分发，并统一裁剪到目标词数。"""
    details = extract_details(image)  # 读图固定 Florence-2
    if story_model["kind"] == "llm":
        story = generate_story_llm(details, story_model["id"], style_desc, words, token)
    else:
        story = generate_story_pipeline(details, style_desc, words, story_model["id"])
    return trim_to_sentence_boundary(story, words)


# ============================================================================ #
# 文本转语音（edge-tts）
# ============================================================================ #
async def _synthesize(text: str, voice: str, rate: str) -> bytes:
    """edge-tts 合成音频，返回 MP3 字节流。"""
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def synthesize(text: str, voice: str, rate: str) -> bytes:
    """同步封装：文本 → MP3 字节。"""
    return asyncio.run(_synthesize(text, voice, rate))


# ============================================================================ #
# UI 渲染
# ============================================================================ #
def inject_css() -> None:
    """注入暖纸质感主题 CSS：纹理底、全直角、无阴影、半透明相框、按钮 hover 高光、动效。"""
    st.markdown(
        """
        <style>
        /* ---- 暖米色纸张噪点纹理底 ---- */
        .stApp {
            background-color: #f2e9d8;
            background-image:
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E"),
                linear-gradient(180deg, rgba(255,252,244,.55), rgba(255,252,244,0) 40%);
        }
        html, body, [class*="css"] {
            font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
        }

        /* ---- 侧边栏：毛玻璃 ---- */
        [data-testid="stSidebar"] {
            background: rgba(255,251,242,.45);
            backdrop-filter: blur(6px);
            border-right: 1px solid rgba(110,86,58,.20);
        }

        /* ---- 标题文字 ---- */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
            color: #3a2e23;
            font-weight: 800;
        }
        .stApp p, .stApp label, .stApp li {
            color: #3a2e23;
        }
        .stApp [data-testid="stCaptionContainer"] {
            color: #8a7a66;
        }

        /* ---- 按钮：扁平直角，hover 顶部高光 ---- */
        [data-testid="stButton"] button {
            border-radius: 0 !important;
            border: none !important;
            background: #c7782e !important;
            color: #fff !important;
            font-weight: 700 !important;
            box-shadow: none !important;
            transition: background .25s ease, box-shadow .25s ease;
        }
        [data-testid="stButton"] button:hover {
            background: #d5893c !important;
            box-shadow: inset 0 2px 0 rgba(255,255,255,.3), inset 0 -2px 0 rgba(0,0,0,.08) !important;
        }

        /* ---- 全局直角：按钮 / 下拉框 / 输入框 / 上传区内部所有元素 ---- */
        [data-testid="stButton"] *,
        [data-testid="stSelectbox"] *,
        [data-testid="stTextInput"] *,
        [data-testid="stFileUploader"] * {
            border-radius: 0 !important;
        }
        /* ---- 输入框 / 下拉框：无阴影、暖白底 ---- */
        [data-testid="stSelectbox"] *,
        [data-testid="stTextInput"] * {
            box-shadow: none !important;
        }
        [data-testid="stSelectbox"] input,
        [data-testid="stTextInput"] input {
            background: #fffdf8 !important;
            border-color: rgba(110,86,58,.25) !important;
        }

        /* ---- 文件上传区：直角虚线框 + 毛玻璃 ---- */
        [data-testid="stFileUploader"] {
            background: rgba(255,251,242,.72) !important;
            backdrop-filter: blur(8px);
        }
        [data-testid="stFileUploader"] section {
            border: 1.5px dashed #c7782e !important;
        }

        /* ---- 暖色半透明相框（预览） ---- */
        .frame {
            border: 8px solid rgba(169,116,60,.65);
            padding: 6px;
            background: rgba(247,236,216,.65);
            margin-bottom: 16px;
            border-radius: 0;
        }
        .frame .preview-inner {
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #d9c9a8, #b9a37e);
            position: relative;
            overflow: hidden;
        }
        .frame .preview-inner .empty {
            color: #8a7a66;
            font-size: 14px;
            text-align: center;
            padding: 24px;
        }
        .frame .preview-inner img {
            width: 100%;
            height: auto;
            display: block;
        }

        /* ---- 故事卡片 ---- */
        .story-card {
            border: 1px solid rgba(110,86,58,.22);
            border-radius: 0;
            padding: 18px 20px;
            margin-bottom: 16px;
            background: rgba(255,251,242,.72);
            backdrop-filter: blur(8px);
            animation: fadeSlideIn .55s cubic-bezier(.22,.61,.36,1);
        }
        .story-card .st-label {
            font-size: 12px;
            font-weight: 700;
            color: #8a7a66;
            margin-bottom: 8px;
        }
        .story-card p {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 15px;
            line-height: 1.8;
            color: #3a2e23;
            margin: 0;
        }
        .story-card .wc {
            font-size: 11px;
            color: #8a7a66;
            margin-top: 10px;
        }

        /* ---- 播放器 ---- */
        .player {
            border: 1px solid rgba(110,86,58,.22);
            border-radius: 0;
            padding: 12px 16px;
            margin-bottom: 16px;
            background: rgba(255,251,242,.72);
            backdrop-filter: blur(8px);
            display: flex;
            align-items: center;
            gap: 14px;
            animation: fadeSlideIn .55s cubic-bezier(.22,.61,.36,1) .08s;
        }
        .player .p-label {
            font-size: 12px;
            font-weight: 700;
            color: #8a7a66;
            white-space: nowrap;
        }
        .player audio { flex: 1; min-width: 0; }

        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_preview_frame(image: Image.Image | None) -> None:
    """渲染暖色相框预览区（未上传时显示占位提示）。"""
    if image is None:
        inner = '<div class="empty">Your image will appear here</div>'
    else:
        preview = resize_image(image, PREVIEW_SIZE)
        b64 = image_to_base64(preview)
        inner = f'<img src="data:image/jpeg;base64,{b64}" alt="uploaded preview">'
    st.markdown(
        f'<div class="frame"><div class="preview-inner">{inner}</div></div>',
        unsafe_allow_html=True,
    )


def render_story_card(story: str) -> None:
    """渲染故事卡片（含词数统计）。"""
    word_count = len(story.split())
    st.markdown(
        f"""
        <div class="story-card">
            <div class="st-label">📖 Story</div>
            <p>{story}</p>
            <div class="wc">{word_count} words</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_audio_player(audio: bytes) -> None:
    """渲染朗读播放器（自定义 HTML audio）。"""
    b64 = base64.b64encode(audio).decode("utf-8")
    st.markdown(
        f"""
        <div class="player">
            <div class="p-label">🔊 Read Aloud</div>
            <audio controls src="data:audio/mp3;base64,{b64}"></audio>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================ #
# 会话状态辅助
# ============================================================================ #
def init_state() -> None:
    """初始化会话状态键。"""
    defaults = {
        "image": None,
        "story": None,
        "audio": None,
        "file_id": None,
        "tts_key": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_upload(uploaded) -> bool:
    """处理上传：新图则加载并重置旧结果。返回是否发生了换图。"""
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.file_id == file_id:
        return False
    st.session_state.file_id = file_id  # 先标记，避免同一文件重复处理
    error = validate_size(uploaded)
    if error:
        st.error(error)
        return False
    st.session_state.image = load_image(uploaded)
    st.session_state.story = None
    st.session_state.audio = None
    st.session_state.tts_key = None
    return True


# ============================================================================ #
# 主流程
# ============================================================================ #
def main() -> None:
    inject_css()
    init_state()

    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("Generation")
        story_model_label = st.selectbox("Story model", list(STORY_MODELS.keys()), filter_mode=None)
        style_label = st.selectbox("Style", list(STORY_STYLES.keys()), filter_mode=None)
        length_label = st.selectbox("Length", list(LENGTHS.keys()), filter_mode=None)

        st.subheader("Voice")
        voice_label = st.selectbox("Voice", list(VOICES.keys()), filter_mode=None)
        speed_label = st.selectbox("Speed", list(SPEEDS.keys()), filter_mode=None)

        # HF Token（仅 LLM 类型模型需要）
        story_model = STORY_MODELS[story_model_label]
        hf_token = ""
        if story_model["kind"] == "llm":
            hf_token = st.text_input(
                "HF Token",
                value=st.secrets.get("HF_TOKEN", ""),
                type="password",
                help="Create a read token at huggingface.co/settings/tokens",
            )

    style_desc = STORY_STYLES[style_label]
    words = LENGTHS[length_label]
    voice = VOICES[voice_label]
    rate = SPEEDS[speed_label]

    # ---------- 主区 ----------
    st.title("🖼️ Image Storyteller")
    st.caption("Upload an image and let AI write a short story from it — then read it aloud.")

    # 1. 预览相框（始终在最上方）
    render_preview_frame(st.session_state.image)

    # 2. 故事卡片 + 播放器（生成后出现，位于相框下方）
    if st.session_state.story:
        # 故事 / 音频解耦：改音色或语速只重跑 TTS，不重跑模型
        tts_key = (voice, rate)
        if st.session_state.tts_key != tts_key or st.session_state.audio is None:
            with st.spinner("Reading aloud..."):
                try:
                    st.session_state.audio = synthesize(st.session_state.story, voice, rate)
                    st.session_state.tts_key = tts_key
                except Exception as e:
                    st.warning(f"Story generated, but audio failed: {e}")

        render_story_card(st.session_state.story)
        if st.session_state.audio:
            render_audio_player(st.session_state.audio)

    # 3. 上传区（始终显示，生成后自然下移垫底）
    uploaded = st.file_uploader(
        "Upload an image",
        type=ALLOWED_TYPES,
        help="PNG / JPG / JPEG / BMP / TIFF · up to 12 MB",
    )
    if uploaded is not None and handle_upload(uploaded):
        st.rerun()  # 换图后立即重跑，避免残留旧内容

    # 4. 生成按钮（始终显示，生成后位于上传区下方）
    button_label = "↻ Regenerate" if st.session_state.story else "✨ Generate Story"
    generate_clicked = st.button(button_label, type="primary")

    if generate_clicked:
        if st.session_state.image is None:
            st.error("Please upload an image first.")
        elif story_model["kind"] == "llm" and not hf_token:
            st.error("Please enter an HF Token in the sidebar.")
        else:
            with st.status("✍️ Writing a story...", expanded=False) as status:
                try:
                    st.session_state.story = generate_story(
                        st.session_state.image,
                        story_model,
                        style_desc,
                        words,
                        hf_token,
                    )
                    st.session_state.tts_key = None  # 强制重新合成
                    status.update(label="✅ Story ready!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Generation failed", state="error", expanded=True)
                    st.error(
                        friendly_hf_error(e)
                        if story_model["kind"] == "llm"
                        else f"Generation failed: {e}"
                    )


if __name__ == "__main__":
    main()
