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
import numpy as np
from transformers import pipeline
from opencc import OpenCC 

# --- 初始化 OpenCC (簡體轉繁體與台灣在地化) ---
cc = OpenCC('s2twp') 

# --- 載入 Hugging Face 情感分析模型 (加上快取，避免每次重整都重新載入) ---
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="tabularisai/multilingual-sentiment-analysis",
        device=-1 # 使用 CPU 進行推論
    )

sentiment_model = load_sentiment_model()

AI_SUMMARY_MODEL_OPTIONS = {
    "原本模型（Llama 3.3 70B Versatile）": "llama-3.3-70b-versatile",
    "Qwen3 32B": "qwen/qwen3-32b",
}

@st.cache_resource
def load_embedding_model():
    return pipeline(
        "feature-extraction",
        model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=-1
    )

embedding_model = load_embedding_model()

# --- 1. 網頁全域配置 ---
st.set_page_config(
    page_title="AI YouTube 輿情分析儀表板", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

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
    keyword = st.text_input("🔍 輸入關鍵字", placeholder="例如：醫美針孔、科技新知...", key="keyword_input")
    search_num = st.slider("📹 分析影片數量", 1, 10, 5, key="search_num_slider")
    st.markdown("---")
    with st.expander("🔑 API 金鑰設定 (必填)"):
        YOUTUBE_API_KEY = st.text_input("YouTube API Key", type="password", key="youtube_api_key_input")
        GROQ_API_KEY = st.text_input("Groq API Key", type="password", key="groq_api_key_input")
    st.markdown("---")
    available_summary_models = list(AI_SUMMARY_MODEL_OPTIONS.keys())
    with st.expander("🧠 AI 文本摘要模型"):
        summary_default_index = 1 if len(available_summary_models) > 1 else 0
        compare_default_index = 0 if len(available_summary_models) > 1 else 0
        summary_model_name = st.selectbox(
            "主摘要模型",
            available_summary_models,
            index=summary_default_index,
            key="summary_model_select",
        )
        compare_model_name = st.selectbox(
            "比較模型",
            available_summary_models,
            index=compare_default_index,
            key="compare_model_select",
        )
    with st.expander("🛡️ Embedding 防網軍參數"):
        sim_threshold = st.slider("文字相似度門檻", 0.70, 0.98, 0.86, 0.01, key="sim_threshold_slider")
        time_window_minutes = st.slider("同步時間窗 (分鐘)", 5, 120, 30, 5, key="time_window_slider")
        risk_threshold = st.slider("可疑分數門檻", 40, 95, 60, 1, key="risk_threshold_slider")
    start_btn = st.button("開始分析", use_container_width=True, key="start_analysis_button")

# --- 3. 核心功能函式 ---

def search_videos(query, max_results=5):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(part="snippet", q=query, type="video", maxResults=max_results)
    response = request.execute()
    
    results = []
    for item in response.get("items", []):
        video_id = item.get("id", {}).get("videoId")
        if video_id:
            results.append({"title": item["snippet"]["title"], "videoId": video_id})
    return results

def get_comments(video_id, max_results=50):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments_data = []
    try:
        request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText")
        response = request.execute()
        
        # 同時抓取留言內容與發布時間
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet.get("textDisplay", "")
            author = snippet.get("authorDisplayName", "未知作者")
            published_at = snippet.get("publishedAt", "")
            # 取出時間字串並只保留 YYYY-MM-DD
            date_str = published_at.split("T")[0]
            if text:
                comments_data.append({
                    "text": text,
                    "date": date_str,
                    "published_at": published_at,
                    "author": author
                })
                
        return comments_data
    except:
        return []

import random # 👉 新增這行匯入隨機套件

def preprocess_comments(comments_data, max_chars=3000):
    seen = set()
    valid_comments = []
    
    for c in comments_data:
        # 將每一則留言統一轉換為台灣繁體
        c_clean = cc.convert(c["text"].strip()) 
        
        if len(c_clean) >= 5 and c_clean not in seen:
            seen.add(c_clean)
            # 保留日期與清理後的文字
            valid_comments.append({
                "text": c_clean,
                "date": c["date"],
                "published_at": c.get("published_at", ""),
                "author": c.get("author", "未知作者")
            })
            
    # --- ✨ 關鍵修正：將留言順序打亂，確保抽樣平均來自各影片 ---
    shuffled_comments = random.sample(valid_comments, len(valid_comments))
    
    llm_text = ""
    for c in shuffled_comments: # 👉 改從洗牌後的清單來抓取
        if len(llm_text) + len(c["text"]) > max_chars:
            break
        llm_text += c["text"] + "\n"
        
    return valid_comments, llm_text

def bert_sentiment(text):
    try:
        # 限制長度避免 BERT 報錯
        result = sentiment_model(text[:512])[0]
        label = result["label"].lower()
        score = float(result["score"])

        if "positive" in label:
            return "正向情緒", score
        elif "negative" in label:
            return "負向情緒", score
        else:
            return "中立/其他", score
    except:
        return "中立/其他", 0.5

def encode_comment_embeddings(texts, batch_size=16):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [t[:256] for t in texts[i:i + batch_size]]
        outputs = embedding_model(batch_texts, truncation=True, padding=True)

        for token_vectors in outputs:
            arr = np.asarray(token_vectors, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]

            if arr.ndim == 2:
                sentence_vec = arr.mean(axis=0)
            elif arr.ndim == 1:
                sentence_vec = arr
            else:
                sentence_vec = np.asarray(arr).reshape(-1)

            norm = np.linalg.norm(sentence_vec)
            if norm > 0:
                sentence_vec = sentence_vec / norm
            vectors.append(sentence_vec)

    return np.vstack(vectors)

def detect_suspicious_comments_embedding(valid_comments, similarity_threshold=0.86, time_window_minutes=30, risk_threshold=60):
    if len(valid_comments) < 3:
        return pd.DataFrame(), pd.DataFrame()

    texts = [c["text"] for c in valid_comments]
    authors = [c.get("author", "未知作者") for c in valid_comments]
    timestamps = pd.to_datetime([c.get("published_at", "") for c in valid_comments], errors="coerce", utc=True)

    embeddings = encode_comment_embeddings(texts)
    if embeddings.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    similarity_matrix = np.matmul(embeddings, embeddings.T)
    author_counts = Counter(authors)
    n = len(valid_comments)

    suspicious_rows = []
    group_ids = [-1] * n
    next_group_id = 1

    for i in range(n):
        similar_neighbors = []
        near_authors = set()

        for j in range(n):
            if i == j:
                continue
            if similarity_matrix[i, j] < similarity_threshold:
                continue

            ti = timestamps[i]
            tj = timestamps[j]
            if pd.isna(ti) or pd.isna(tj):
                continue

            delta_min = abs((ti - tj).total_seconds()) / 60.0
            if delta_min <= time_window_minutes:
                similar_neighbors.append(j)
                near_authors.add(authors[j])

        if not similar_neighbors:
            continue

        if group_ids[i] == -1:
            group_ids[i] = next_group_id
            next_group_id += 1
        for j in similar_neighbors:
            if group_ids[j] == -1:
                group_ids[j] = group_ids[i]

        similarity_score = min(len(similar_neighbors) / 5.0, 1.0) * 60.0
        coordination_score = min(len(near_authors) / 4.0, 1.0) * 25.0
        repeated_author_score = min(max(author_counts[authors[i]] - 1, 0) / 4.0, 1.0) * 15.0
        risk_score = round(similarity_score + coordination_score + repeated_author_score, 1)

        if risk_score >= risk_threshold:
            suspicious_rows.append({
                "群組ID": group_ids[i],
                "作者": authors[i],
                "發布時間": str(timestamps[i]) if not pd.isna(timestamps[i]) else "未知",
                "留言": texts[i],
                "可疑分數": risk_score,
                "相似鄰居數": len(similar_neighbors),
                "近鄰作者數": len(near_authors)
            })

    if not suspicious_rows:
        return pd.DataFrame(), pd.DataFrame()

    suspicious_comments_df = pd.DataFrame(suspicious_rows).drop_duplicates(subset=["作者", "留言"])
    suspicious_comments_df = suspicious_comments_df.sort_values(by="可疑分數", ascending=False).reset_index(drop=True)

    suspicious_authors_df = (
        suspicious_comments_df
        .groupby("作者", as_index=False)
        .agg(可疑留言數=("留言", "count"), 平均可疑分數=("可疑分數", "mean"), 最高可疑分數=("可疑分數", "max"))
        .sort_values(by=["可疑留言數", "最高可疑分數"], ascending=False)
        .reset_index(drop=True)
    )
    suspicious_authors_df["平均可疑分數"] = suspicious_authors_df["平均可疑分數"].round(1)

    return suspicious_comments_df, suspicious_authors_df

def rebuild_llm_text(comments_data, max_chars=3000):
    llm_text = ""
    for c in comments_data:
        if len(llm_text) + len(c["text"]) > max_chars:
            break
        llm_text += c["text"] + "\n"
    return llm_text

# （將原本的 analyze_sentiment_and_trend 與 extract_representative_comments 刪除）
# （用下面這個合併優化版的 analyze_all_sentiment_data 取代，讓 BERT 只需算一次！）
def analyze_all_sentiment_data(valid_comments):
    sentiments = []
    trend_data = []
    rows = []
    
    for c in valid_comments:
        text = c["text"]
        date = c["date"]
        
        # BERT 模型推論 (每則留言只做一次，大幅節省時間)
        label, score = bert_sentiment(text)
            
        sentiments.append(label)
        trend_data.append({"日期": date, "情緒": label, "數量": 1})
        rows.append({
            "留言範例": text,
            "模型信心": round(score * 100, 2),
            "情感標籤": label
        })
        
    # 1. 整理圓餅圖資料
    overall_counts = Counter(sentiments)
    
    # 2. 整理趨勢圖資料
    df_trend = pd.DataFrame(trend_data)
    if not df_trend.empty:
        df_trend = df_trend.groupby(["日期", "情緒"]).sum().reset_index()
        df_trend = df_trend.sort_values(by="日期")
        
    # 3. 整理 Top 5 正反面留言 (加入 70% 信心度門檻)
    df_all = pd.DataFrame(rows)
    positive_df = pd.DataFrame()
    negative_df = pd.DataFrame()

    if not df_all.empty:
        positive_df = df_all[
            (df_all["情感標籤"] == "正向情緒") & (df_all["模型信心"] > 70.0)
        ].sort_values(by="模型信心", ascending=False).head(5).reset_index(drop=True)
        
        negative_df = df_all[
            (df_all["情感標籤"] == "負向情緒") & (df_all["模型信心"] > 70.0)
        ].sort_values(by="模型信心", ascending=False).head(5).reset_index(drop=True)

    return overall_counts, df_trend, positive_df, negative_df

def get_ai_comprehensive_analysis(text, model_id):
    if not text:
        return {"error": "⚠️ 沒有足夠的有效留言可供分析。"}
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""
        你是一位專業的社群媒體輿情分析師。請根據以下 YouTube 留言，進行「深度且詳細」的綜合分析。
        請嚴格輸出為 JSON 格式，包含以下兩個 key。
        ⚠️ 重要指示：請勿給出過於簡短的敷衍回應！每個標題下的內容都需要有實質的分析、具體的觀點整理，並使用「繁體中文」與「Markdown 格式」排版。

        期望的 JSON 結構與內容深度要求如下：
        {{
            "summary": "📌 **【事件簡介】**：\\n(請詳細說明討論的核心事件，約 100 字)\\n\\n🔄 **【事件經過與背景】**：\\n(根據留言整理發展脈絡，約 100-150 字)\\n\\n⚡ **【主要爭議點】**：\\n(網友最在意的焦點是什麼？約 150 字)",
            "opinion": "👍 **【支持方論點】**：\\n(贊同的網友主要理由與邏輯是什麼？約 200 字)\\n\\n👎 **【反對方論點】**：\\n(批評的反對意見核心不滿在哪裡？約 200 字)\\n\\n🧭 **【整體風向】**：\\n(深入剖析輿論大方向，約 150 字)\\n\\n🎭 **【情緒特徵】**：\\n(留言中充斥著什麼樣的情緒？約 100 字)"
        }}

        以下為留言內容：
        \"\"\"
        {text}
        \"\"\"
        """
        res = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是一個只會輸出合法 JSON 格式，且擅長進行長篇深度分析的輿情助理。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, 
            max_tokens=2500,
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
        if "taiwan" in font.name.lower() or  "tc" in font.name.lower() or "microsoft" in font.name.lower():
            return font.fname
    return None

# --- 4. 主畫面邏輯 (Main Content) ---

# 改用單純的 if start_btn，讓缺少條件時可以跳出警告，而不是沒反應
if start_btn:
    if not keyword:
        st.warning("⚠️ 請先在左側控制面板輸入「關鍵字」再開始分析！")
    elif not YOUTUBE_API_KEY or not GROQ_API_KEY:
        st.error("❌ 請展開左側的「API 金鑰設定」並輸入正確的金鑰！")
    else:
        with st.spinner('🎯 正在擷取資料、推論模型並進行深度分析中，這可能需要一點時間...'):
            videos = search_videos(keyword, max_results=search_num)
            all_comments = []
            for v in videos:
                all_comments.extend(get_comments(v['videoId']))
            
            if not all_comments:
                st.error("😔 找不到相關留言，或該影片的評論功能已被關閉。")
            else:
                valid_comments, llm_text = preprocess_comments(all_comments)
                suspicious_comments_df, suspicious_authors_df = detect_suspicious_comments_embedding(
                    valid_comments,
                    similarity_threshold=sim_threshold,
                    time_window_minutes=time_window_minutes,
                    risk_threshold=risk_threshold
                )

                suspicious_keys = set(
                    zip(
                        suspicious_comments_df.get("作者", pd.Series(dtype=str)),
                        suspicious_comments_df.get("留言", pd.Series(dtype=str))
                    )
                ) if not suspicious_comments_df.empty else set()
                filtered_comments = [
                    c for c in valid_comments
                    if (c.get("author", "未知作者"), c.get("text", "")) not in suspicious_keys
                ]
                cleaned_llm_text = rebuild_llm_text(filtered_comments)
                summary_model_id = AI_SUMMARY_MODEL_OPTIONS[summary_model_name]
                compare_model_id = AI_SUMMARY_MODEL_OPTIONS[compare_model_name]
                
                st.markdown(f"## 📊 輿情分析報告: **{keyword}**")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric(label="🎬 觀測影片", value=f"{len(videos)} 部")
                with m2:
                    st.metric(label="💬 原始擷取留言", value=f"{len(all_comments)} 則")
                with m3:
                    st.metric(label="✨ 過濾後有效留言", value=f"{len(filtered_comments)} 則", help="已過濾重複、過短與可疑協同行為留言")
                
                st.markdown("---")
                
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📋 AI 文本摘要",
                    "📊 情感分佈",
                    "⏳ 時間軸趨勢",
                    "☁️ 詞雲熱詞",
                    "💬 原始數據"
                ])
                
                ai_result = get_ai_comprehensive_analysis(cleaned_llm_text, summary_model_id)
                compare_ai_result = None
                if compare_model_name != summary_model_name:
                    compare_ai_result = get_ai_comprehensive_analysis(cleaned_llm_text, compare_model_id)
                
                with tab1:
                    if "error" in ai_result:
                        st.error(ai_result["error"])
                    else:
                        st.caption(f"主摘要模型：{summary_model_name}；比較模型：{compare_model_name}")
                        col_s1, col_s2 = st.columns(2)
                        with col_s1:
                            st.markdown('<div class="report-card">', unsafe_allow_html=True)
                            st.caption(summary_model_name)
                            st.subheader("📝 事件背景與核心脈絡")
                            st.markdown(ai_result.get("summary", "無法解析內容"))
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col_s2:
                            st.markdown('<div class="report-card">', unsafe_allow_html=True)
                            st.caption(summary_model_name)
                            st.subheader("⚖️ 網友正反觀點碰撞")
                            st.markdown(ai_result.get("opinion", "無法解析內容"))
                            st.markdown('</div>', unsafe_allow_html=True)

                        if compare_ai_result:
                            st.markdown("---")
                            st.subheader("🔍 模型比較")
                            if "error" in compare_ai_result:
                                st.warning(f"{compare_model_name} 輸出失敗：{compare_ai_result['error']}")
                            else:
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                                    st.caption(compare_model_name)
                                    st.subheader("📝 事件背景與核心脈絡")
                                    st.markdown(compare_ai_result.get("summary", "無法解析內容"))
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with c2:
                                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                                    st.caption(compare_model_name)
                                    st.subheader("⚖️ 網友正反觀點碰撞")
                                    st.markdown(compare_ai_result.get("opinion", "無法解析內容"))
                                    st.markdown('</div>', unsafe_allow_html=True)

                # 呼叫合併優化後的情感分析函式，一次拿回四個資料！
                sentiment_counts, df_trend, positive_df, negative_df = analyze_all_sentiment_data(filtered_comments)

                with tab2:
                    st.subheader("📊 留言情緒比例分析")
                    s_df = pd.DataFrame(sentiment_counts.items(), columns=['情緒', '數量'])
                    fig_pie = px.pie(s_df, values='數量', names='情緒', color='情緒',
                                     color_discrete_map={'正向情緒':'#2ecc71', '負向情緒':'#e74c3c', '中立/其他':'#bdc3c7'},
                                     hole=0.4)
                    
                    is_dark = st.get_option("theme.base") == "dark"
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white" if is_dark else "black")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.markdown("---")
                    st.subheader("📝 正負面代表性留言 (Top 5，信心度 > 70%)")
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.markdown("#### ✅ 正向留言")
                        if not positive_df.empty:
                            st.dataframe(positive_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("目前沒有抓到信心度大於 70% 的正面留言。")
                    with col_right:
                        st.markdown("#### ❌ 負向留言")
                        if not negative_df.empty:
                            st.dataframe(negative_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("目前沒有抓到信心度大於 70% 的負面留言。")

                with tab3:
                    st.subheader("📈 每日留言情緒走勢圖")
                    if not df_trend.empty:
                        fig_trend = px.line(df_trend, x="日期", y="數量", color="情緒", markers=True,
                                            color_discrete_map={'正向情緒':'#2ecc71', '負向情緒':'#e74c3c', '中立/其他':'#bdc3c7'},
                                            title="情緒聲量時間軸變化")
                        
                        fig_trend.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)', 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            font_color="white" if is_dark else "black",
                            xaxis_title="日期",
                            yaxis_title="留言聲量 (則數)"
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("無法繪製趨勢圖：留言時間數據不足。")

                with tab4:
                    st.subheader("☁️ 熱門關鍵字雲")
                    words = jieba.cut(" ".join([c["text"] for c in filtered_comments]))
                    stopwords = {
                        "的", "是", "我", "了", "啊", "嗎", "吧", "也", "就", "都", "很", "還", "在", "有", "和", "不", "人", "他", "你", "這", "那", 
                        "影片", "留言", "覺得", "真的", "怎麼", "什麼", "看到", "知道", "出來", "認為",
                        "就是", "我們", "他們", "這個", "可以", "只是", "還是", "那些", "那麼", "因為", "所以", "如果", "但是", 
                        "一樣", "一個", "這樣", "現在", "其實", "自己", "這些", "時候", "沒有", "不是", "不過", "的話", "大家", 
                        "而且", "這麼", "為什麼", "一直", "已經", "可能", "應該", "然後", "哪怕", "哪怕", "哪怕", "甚至", "這種",
                        "那些", "有些", "為何", "到底", "多少", "一些", "很多", "這麼", "這麼", "的話", "不會", "不能", "不要", "完全", "https"
                    }
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

                with tab5:
                    st.subheader("💬 擷取之原始留言列表")
                    display_df = pd.DataFrame(filtered_comments)
                    display_df.rename(columns={
                        "published_at": "發布時間",
                        "date": "發布日期",
                        "author": "作者",
                        "text": "留言內容"
                    }, inplace=True)
                    display_df = display_df[["發布日期", "發布時間", "作者", "留言內容"]]
                    st.dataframe(display_df, use_container_width=True)

# 當按鈕沒有被按下時，顯示歡迎畫面
else:
    st.markdown("""
        <div class="welcome-card">
            <h1>👋 歡迎使用 AI YouTube 輿情分析系統</h1>
            <p style="font-size: 18px; opacity: 0.9;">
                本系統結合了 YouTube Data API、Hugging Face BERT 情感模型與 Groq Llama3，能高效爬取特定主題的社群留言，並自動產出脈絡摘要、觀點碰撞、情感分佈與動態時間軸趨勢。
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