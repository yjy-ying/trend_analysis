import streamlit as st
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from groq import Groq
from collections import Counter
import jieba
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import json
from snownlp import SnowNLP
from opencc import OpenCC

# --- 1. 網頁全域配置 ---
st.set_page_config(
    page_title="AI YouTube 輿情分析儀表板", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# 支援深淺色切換的客製化高級感 CSS
st.markdown("""
    <style>
    .welcome-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 40px;
        border-radius: 16px;
        color: white !important;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 30px;
    }
    .welcome-card h1, .welcome-card p { color: #ffffff !important; }
    .step-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .step-box h4 { color: var(--text-color) !important; margin-bottom: 5px; }
    .report-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-top: 4px solid #764ba2;
    }
    .report-card h3 { color: var(--text-color) !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 側邊欄設定 (Sidebar) ---
with st.sidebar:
    st.title("⚙️ 控制面板")
    st.markdown("---")
    keyword = st.text_input("🔍 輸入關鍵字", placeholder="例如：醫美針孔、科技新知...")
    search_num = st.slider("📹 分析影片數量", 1, 10, 5)
    st.markdown("---")
    with st.expander("🔑 API 金鑰設定 (必填)"):
        YOUTUBE_API_KEY = st.text_input("YouTube API Key", type="password")
        GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    st.markdown("---")
    start_btn = st.button("開始分析", use_container_width=True)

# --- 3. 核心功能函式 ---

def search_videos(query, max_results=5):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(part="snippet", q=query, type="video", maxResults=max_results)
    response = request.execute()
    
    results = []
    # 遍歷回傳的 items，並使用安全的 .get() 方法避免 KeyError
    for item in response.get("items", []):
        # 嘗試抓取 videoId，如果沒有這個 key 則回傳 None 而不是報錯
        video_id = item.get("id", {}).get("videoId")
        
        # 只有在真的有 videoId 的情況下，才加入清單中
        if video_id:
            results.append({
                "title": item["snippet"]["title"], 
                "videoId": video_id
            })
            
    return results
def get_comments(video_id, max_results=50):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText")
        response = request.execute()
        return [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]]
    except:
        return []
    
cc = OpenCC('s2twp')

# 【優化 3】降噪與前處理：幫留言減肥
def preprocess_comments(comments, max_chars=3000):
    seen = set()
    valid_comments = []
    
    # 1. 簡繁轉換、去除重複洗版與過短廢話
    for c in comments:
        # 將每一則留言統一轉換為台灣繁體與慣用語
        c_clean = cc.convert(c.strip()) 
        
        # 剔除小於 5 個字的留言，並過濾重複
        if len(c_clean) >= 5 and c_clean not in seen:
            seen.add(c_clean)
            valid_comments.append(c_clean)
            
    # 2. 限制總字元數 (避免爆 Token)
    llm_text = ""
    for c in valid_comments:
        if len(llm_text) + len(c) > max_chars:
            break
        llm_text += c + "\n"
        
    return valid_comments, llm_text

# 【優化 2】本地端情感分析：脫離 LLM，省下海量 Token
def analyze_sentiment_local(comments):
    sentiments = []
    for c in comments:
        try:
            # SnowNLP 情緒分數落在 0~1 之間，越接近 1 越正向
            s = SnowNLP(c).sentiments
            if s > 0.65:
                sentiments.append("正向情緒")
            elif s < 0.35:
                sentiments.append("負向情緒")
            else:
                sentiments.append("中立/其他")
        except:
            sentiments.append("中立/其他")
    return Counter(sentiments)

# 【優化 1】合併 API 請求：一次拿完摘要與觀點，並強制要求深度分析
def get_ai_comprehensive_analysis(text):
    if not text:
        return {"error": "⚠️ 沒有足夠的有效留言可供分析。"}
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # 重新設計的 Prompt：嚴格要求字數與分析深度，並加入換行符號 (\n\n) 讓排版更美觀
        prompt = f"""
        你是一位專業的社群媒體輿情分析師。請根據以下 YouTube 留言，進行「深度且詳細」的綜合分析，可對於部分內容列點展示。
        請嚴格輸出為 JSON 格式，包含以下兩個 key。
        ⚠️ 重要指示：請勿給出過於簡短的敷衍回應！每個標題下的內容都需要有實質的分析、具體的觀點整理，並使用「繁體中文」與「Markdown 格式」排版。

        期望的 JSON 結構與內容深度要求如下：
        {{
            "summary": "📌 **【事件簡介】**：\\n(請詳細說明這群留言主要在討論什麼核心事件，約 100 字)\\n\\n🔄 **【事件經過與背景】**：\\n(根據留言透露的訊息，詳細整理這個事件的發展脈絡，約 100-150 字)\\n\\n⚡ **【主要爭議點】**：\\n(網友們最在意、爭論最激烈的焦點是什麼？請列出具體細節，約 150 字)",
            "opinion": "👍 **【支持方論點】**：\\n(贊同、支持的網友，他們的主要理由與邏輯是什麼？請詳細列出，約 150 字)\\n\\n👎 **【反對方論點】**：\\n(質疑、批評的反對意見，他們的核心不滿在哪裡？請詳細列出，約 150 字)\\n\\n🧭 **【整體風向】**：\\n(深入剖析社群上的輿論大方向，例如是一面倒、五五波還是各有堅持，約 100 字)\\n\\n🎭 **【情緒特徵】**：\\n(留言中充斥著什麼樣的情緒？請具體說明，約 100 字)"
        }}

        以下為留言內容：
        \"\"\"
        {text}
        \"\"\"
        """
        
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "你是一個只會輸出合法 JSON 格式，且擅長進行長篇深度分析的輿情助理。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # 稍微調高一點點溫度，讓文字更豐富生動，原本 0.0 會太死板
            max_tokens=2500, # 將上限從 1500 提高到 2500，確保長篇大論不會被截斷
            response_format={"type": "json_object"} 
        )
        content = res.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        return {"error": f"❌ AI 分析過程中發生錯誤：{str(e)}"}

def get_windows_font():
    paths = ["C:\\Windows\\Fonts\\msjh.ttc", "C:\\Windows\\Fonts\\msjhbd.ttc", "C:\\Windows\\Fonts\\simsun.ttc"]
    for p in paths:
        if os.path.exists(p): return p
    for font in fm.fontManager.ttflist:
        if "taiwan" in font.name.lower() or "tc" in font.name.lower() or "microsoft" in font.name.lower():
            return font.fname
    return None

# --- 4. 主畫面邏輯 (Main Content) ---
if start_btn and keyword:
    if not YOUTUBE_API_KEY or not GROQ_API_KEY:
        st.error("❌ 請展開左側的「API 金鑰設定」並輸入正確的金鑰！")
        st.stop()
        
    with st.spinner('🎯 正在擷取資料並進行多維度分析中，請稍候...'):
        videos = search_videos(keyword, max_results=search_num)
        all_comments = []
        for v in videos:
            all_comments.extend(get_comments(v['videoId']))
        
        if not all_comments:
            st.error("😔 找不到相關留言，或該影片的評論功能已被關閉。")
        else:
            # 進行資料前處理
            valid_comments, llm_text = preprocess_comments(all_comments)
            
            st.markdown(f"## 📊 輿情分析報告: **{keyword}**")
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="🎬 觀測影片", value=f"{len(videos)} 部")
            with m2:
                st.metric(label="💬 原始擷取留言", value=f"{len(all_comments)} 則")
            with m3:
                st.metric(label="✨ 過濾後有效留言", value=f"{len(valid_comments)} 則", help="已過濾重複與過短的無意義留言")
            
            st.markdown("---")
            
            # --- 分頁配置 ---
            tab1, tab2, tab3, tab4 = st.tabs(["📋 AI 文本摘要", "📈 情感分佈", "☁️ 詞雲熱詞", "💬 原始數據"])
            
            # 呼叫大模型 (僅需呼叫一次！)
            ai_result = get_ai_comprehensive_analysis(llm_text)
            
            with tab1:
                if "error" in ai_result:
                    st.error(ai_result["error"])
                else:
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.subheader("📝 事件背景與核心脈絡")
                        st.markdown(ai_result.get("summary", "無法解析內容"))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_s2:
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.subheader("⚖️ 網友正反觀點碰撞")
                        st.markdown(ai_result.get("opinion", "無法解析內容"))
                        st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.subheader("📊 留言情緒比例分析")
                # 使用本地端套件分析情緒，零 Token 消耗，且分析數量為「所有有效留言」而非截斷後的留言
                sentiment_counts = analyze_sentiment_local(valid_comments)
                
                s_df = pd.DataFrame(sentiment_counts.items(), columns=['情緒', '數量'])
                
                fig = px.pie(s_df, values='數量', names='情緒', color='情緒',
                             color_discrete_map={'正向情緒':'#2ecc71', '負向情緒':'#e74c3c', '中立/其他':'#bdc3c7'},
                             hole=0.4)
                
                is_dark = st.get_option("theme.base") == "dark"
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="white" if is_dark else "black"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("☁️ 熱門關鍵字雲")
                words = jieba.cut(" ".join(valid_comments))
                
                # 🛑 大幅擴充停用詞庫（Stopwords）
                stopwords = {
                    # 單字
                    "的", "是", "我", "了", "啊", "嗎", "吧", "也", "就", "都", "很", "還", "在", "有", "和", "不", "人", "他", "你", "這", "那", 
                    # 影片相關與常見動詞/形容詞
                    "影片", "留言", "覺得", "真的", "怎麼", "什麼", "看到", "知道", "出來", "認為",
                    # 常見代名詞、連詞、副詞與無意義口頭禪 (針對詞雲優化)
                    "就是", "我們", "他們", "這個", "可以", "只是", "還是", "那些", "那麼", "因為", "所以", "如果", "但是", 
                    "一樣", "一個", "這樣", "現在", "其實", "自己", "這些", "時候", "沒有", "不是", "不過", "的話", "大家", 
                    "而且", "這麼", "為什麼", "一直", "已經", "可能", "應該", "然後", "哪怕", "哪怕", "哪怕", "甚至", "這種",
                    "那些", "有些", "為何", "到底", "多少", "一些", "很多", "這麼", "這麼", "的話", "不會", "不能", "不要"
                }
                
                # 過濾掉停用詞，並且強制詞長大於 1
                filtered = [w for w in words if w.strip() not in stopwords and len(w) > 1]
                
                if len(filtered) > 0:
                    font_p = get_windows_font()
                    try:
                        bg_color = "#1e1e1e" if st.get_option("theme.base") == "dark" else "white"
                        wc_args = {"background_color": bg_color, "width": 1000, "height": 450}
                        if font_p: wc_args["font_path"] = font_p
                            
                        wc = WordCloud(**wc_args).generate(" ".join(filtered))
                        fig_wc, ax = plt.subplots(figsize=(10, 4.5), facecolor=bg_color)
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc)
                    except:
                        st.info("💡 詞雲生成發生字體相容性錯誤。")
                else:
                    st.info("暫無足夠的關鍵字可以生成詞雲。")

            with tab4:
                st.subheader("💬 擷取之原始留言列表")
                st.dataframe(pd.DataFrame(all_comments, columns=["留言內容"]), use_container_width=True)

else:
    # --- 5. 預設歡迎畫面 ---
    st.markdown("""
        <div class="welcome-card">
            <h1>👋 歡迎使用 AI YouTube 輿情分析系統</h1>
            <p style="font-size: 18px; opacity: 0.9;">
                本系統結合了 YouTube Data API、SnowNLP 與 Groq Llama3，能高效爬取特定主題的社群留言，並自動產出脈絡摘要、觀點碰撞與情感分佈圖表。
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("💡 快速開始三步驟：")
    st.markdown("""
        <div class="step-box">
            <h4>1️⃣ 設定 API 金鑰</h4>
            <p>展開左側側邊欄的 <b>「API 金鑰設定」</b>，填入您的 YouTube 與 Groq API Key。</p>
        </div>
        <div class="step-box">
            <h4>2️⃣ 輸入想分析的關鍵字</h4>
            <p>在左側欄輸入框填入您感興趣的議題或事件關鍵字。</p>
        </div>
        <div class="step-box">
            <h4>3️⃣ 啟動分析</h4>
            <p>點擊 <b>「啟動分析」</b> 按鈕，稍等片刻！</p>
        </div>
        """, unsafe_allow_html=True)