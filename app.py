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
    /* 歡迎區區塊樣式 (深淺色模式下皆保持極具質感的藍色漸層) */
    .welcome-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 40px;
        border-radius: 16px;
        color: white !important;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 30px;
    }
    .welcome-card h1, .welcome-card p { color: #ffffff !important; }
    
    /* 步驟說明卡片：使用系統內建變數，自動調適底色與字體顏色 */
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
    
    /* 報告區塊樣式：自動調適底色與字體顏色，防止深色模式下字體隱形 */
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
    
    # 搜尋參數
    keyword = st.text_input("🔍 輸入關鍵字", placeholder="例如：醫美針孔、科技新知...")
    search_num = st.slider("📹 分析影片數量", 1, 10, 5)
    
    st.markdown("---")
    # 將 API 金鑰收入摺疊選單中，保持介面整潔
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
    return [{"title": item["snippet"]["title"], "videoId": item["id"]["videoId"]} for item in response["items"]]

def get_comments(video_id, max_results=50):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText")
        response = request.execute()
        return [item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] for item in response["items"]]
    except:
        return []

def get_ai_analysis(comments, prompt_type):
    # 安全機制：攔截無有效留言的狀況
    if not comments or len(comments) == 0:
        return "⚠️ 目前沒有擷取到任何有效的留言，無法進行 AI 輿情分析。"
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        clean_comments = [c.strip() for c in comments if c and c.strip()]
        text = "\n".join(clean_comments[:80])
        
        if not text:
            return "⚠️ 擷取到的留言皆為空值，無法進行 AI 分析。"

        # 根據不同的分析要求給予進階結構化的 Prompt 導引
        if prompt_type == "summary":
            prompt = f"""
            你是一位專業的社群媒體輿情分析師。請根據以下提供的 YouTube 影片留言，整理並產出一份精煉的網路事件分析報告。
            報告必須包含以下三個區塊：
            1. 📌【事件簡介】：這群留言主要在討論什麼主題或核心事件？
            2. 🔄【事件經過/背景】：根據留言透露的訊息，這個事件的發展脈絡為何？
            3. ⚡【主要爭議點】：網友們最在意、爭論最激烈的焦點是什麼？

            請嚴格使用「繁體中文（台灣）」與「Markdown 格式」進行回答，字數精煉、條理分明。
            
            以下為留言內容：
            \"\"\"
            {text}
            \"\"\"
            """
        else:
            prompt = f"""
            你是一位專業的社群媒體輿情分析師。請根據以下提供的 YouTube 影片留言，深度分析整體的網路風向與正反方觀點。
            報告必須包含以下四個區塊：
            1. 👍【支持方論點】：贊同、支持影片觀點或事件主軸的網友，他們的主要理由是什麼？
            2. 👎【反對方論點】：質疑、批評或持反對意見的網友，他們的核心不滿在哪裡？
            3. 🧭【整體風向】：目前社群上的輿論大方向是偏向哪一方？（例如：一面倒批評、五五波爭論、多數支持等）
            4. 🎭【情緒特徵】：留言中普遍充斥著什麼樣的情緒？（例如：憤怒、嘲諷、焦慮、理性討論等）

            請嚴格使用「繁體中文（台灣）」與「Markdown 格式」進行回答，字數精煉、條理分明。

            以下為留言內容：
            \"\"\"
            {text}
            \"\"\"
            """
        
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "你是一個只會使用繁體中文（台灣）回答的專業輿情分析助理。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # 降低隨機性，確保結構穩定
            max_tokens=1500
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ AI 分析過程中發生錯誤：{str(e)}"

def get_windows_font():
    # 自動尋找 Windows 系統可用的中文字體路徑，防止亂碼豆腐塊
    paths = ["C:\\Windows\\Fonts\\msjh.ttc", "C:\\Windows\\Fonts\\msjhbd.ttc", "C:\\Windows\\Fonts\\simsun.ttc"]
    for p in paths:
        if os.path.exists(p): 
            return p
    for font in fm.fontManager.ttflist:
        if "taiwan" in font.name.lower() or "tc" in font.name.lower() or "microsoft" in font.name.lower():
            return font.fname
    return None

# --- 4. 主畫面邏輯 (Main Content) ---

# 檢查是否按下按鈕且有輸入關鍵字
if start_btn and keyword:
    # 檢查金鑰是否存在
    if not YOUTUBE_API_KEY or not GROQ_API_KEY:
        st.error("❌ 請展開左側的「API 金鑰設定」並輸入正確的金鑰！")
        st.stop()
        
    with st.spinner('🎯 正在從 YouTube 擷取資料並請 AI 進行深度分析中，請稍候...'):
        # 抓取資料
        videos = search_videos(keyword, max_results=search_num)
        all_comments = []
        for v in videos:
            all_comments.extend(get_comments(v['videoId']))
        
        if not all_comments:
            st.error("😔 找不到相關留言，或該影片的評論功能已被關閉。")
        else:
            # 數據儀表板核心標題
            st.markdown(f"## 📊 輿情分析報告: **{keyword}**")
            
            # 頂部數據小字卡 (Metrics)
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label="🎬 觀測影片數量", value=f"{len(videos)} 部")
            with m2:
                st.metric(label="💬 擷取留言總數", value=f"{len(all_comments)} 則")
            
            st.markdown("---")
            
            # --- 分頁配置 ---
            tab1, tab2, tab3, tab4 = st.tabs(["📋 AI 文本摘要", "📈 情感分佈", "☁️ 詞雲熱詞", "💬 原始數據"])
            
            with tab1:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.subheader("📝 事件背景與核心脈絡")
                    # 呼叫整合優化後的 AI 分析，並使用 Markdown 漂亮呈現
                    summary = get_ai_analysis(all_comments, "summary")
                    st.markdown(summary)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_s2:
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.subheader("⚖️ 網友正反觀點碰撞")
                    opinion = get_ai_analysis(all_comments, "opinion")
                    st.markdown(opinion)
                    st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.subheader("📊 留言情緒比例分析")
                pos_words = ["好","喜歡","讚","支持","不錯","可以","有效","推薦","值得","安心","安全","漂亮","好看","專業"]
                neg_words = ["爛","差","失望","噁心","不行","危險","可怕","後悔","痛","貴","騙","怪","沒效","失敗"]
                
                sentiments = []
                for c in all_comments:
                    # 改用計分制，有效改善極端比例問題
                    pos_score = sum(1 for w in pos_words if w in c)
                    neg_score = sum(1 for w in neg_words if w in c)
                    
                    if pos_score > neg_score:
                        sentiments.append("正向情緒")
                    elif neg_score > pos_score:
                        sentiments.append("負向情緒")
                    else:
                        sentiments.append("中立/其他")
                
                s_df = pd.DataFrame(Counter(sentiments).items(), columns=['情緒', '數量'])
                
                # 建構圓餅圖
                fig = px.pie(s_df, values='數量', names='情緒', color='情緒',
                             color_discrete_map={'正向情緒':'#2ecc71', '負向情緒':'#e74c3c', '中立/其他':'#bdc3c7'},
                             hole=0.4)
                
                # 讓 Plotly 圖表的文字配色相容 Streamlit 的深色/淺色主題
                is_dark = st.get_option("theme.base") == "dark"
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="white" if is_dark else "black"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("☁️ 熱門關鍵字雲")
                words = jieba.cut(" ".join(all_comments))
                # 擴大停用字清單，大幅提升詞雲分析價值
                stopwords = {"的", "是", "我", "了", "啊", "嗎", "吧", "也", "就", "都", "很", "還", "在", "有", "和", "不", "人", "他", "你", "這", "那", "影片", "留言", "覺得", "真的"}
                filtered = [w for w in words if w.strip() not in stopwords and len(w) > 1]
                
                if len(filtered) > 0:
                    font_p = get_windows_font()
                    try:
                        # 詞雲底色自動配合網頁主題切換
                        bg_color = "#1e1e1e" if st.get_option("theme.base") == "dark" else "white"
                        
                        if font_p:
                            wc = WordCloud(background_color=bg_color, width=1000, height=450, font_path=font_p).generate(" ".join(filtered))
                        else:
                            wc = WordCloud(background_color=bg_color, width=1000, height=450).generate(" ".join(filtered))
                        
                        fig_wc, ax = plt.subplots(figsize=(10, 4.5), facecolor=bg_color)
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc)
                    except:
                        st.info("💡 詞雲生成時發生字體相容性錯誤，但您可以參考前方的 AI 文本摘要。")
                else:
                    st.info("暫無足夠的關鍵字可以生成詞雲。")

            with tab4:
                st.subheader("💬 擷取之原始留言列表")
                st.dataframe(pd.DataFrame(all_comments, columns=["留言內容"]), use_container_width=True)

else:
    # --- 5. 預設歡迎畫面 (初始引導頁面) ---
    st.markdown("""
        <div class="welcome-card">
            <h1>👋 歡迎使用 AI YouTube 輿情分析系統</h1>
            <p style="font-size: 18px; opacity: 0.9;">
                本系統結合了 YouTube Data API 與 Groq Llama3 大型語言模型，能一鍵幫您爬取特定主題的社群留言，並自動產出脈絡摘要、觀點碰撞與情感分佈圖表。
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("💡 快速開始三步驟：")
    
    st.markdown("""
        <div class="step-box">
            <h4>1️⃣ 設定 API 金鑰</h4>
            <p>展開左側側邊欄的 <b>「API 金鑰設定」</b>，填入您的 YouTube API Key 與 Groq API Key。</p>
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