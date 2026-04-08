import streamlit as st
import pandas as pd
from openai import OpenAI
import io

# === 🌟 可视化相关库 🌟 ===
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
import re
import plotly.express as px  # 新增 Plotly 库

# ==========================================
# 0. 全局美化设置 & NLTK 初始化
# ==========================================
plt.style.use('dark_background')
plt.rcParams.update({
    "figure.facecolor": (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
})

try:
    nltk.download('stopwords', quiet=True)
except Exception:
    st.error("NLTK 停用词下载失败，词云图效果可能不佳。请检查网络。")

# ==========================================
# 1. 页面基础配置
# ==========================================
st.set_page_config(
    page_title="穿针引线跨境电商 | 全维度VOC智能助手",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button { 
        width: 100%; 
        background-color: #007bff; 
        color: white; 
        border-radius: 8px; 
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: #0056b3; transform: translateY(-2px); }
    h1 { font-weight: 800; color: #E0E0E0; }
    h3 { font-weight: 600; color: #4DA8DA; }
    /* 让 Plotly 图标适配暗黑模式 */
    .js-plotly-plot .plotly .modebar { left: 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 亚马逊全维度 VOC 深度洞察助手")
st.markdown("---")

# ==========================================
# 2. 侧边栏配置
# ==========================================
with st.sidebar:
    st.header("🔑 配置中心")
    api_key = st.text_input("输入 API Key", type="password")
    base_url = st.selectbox("选择供应商", ["https://api.deepseek.com", "https://api.openai.com/v1"], index=0)
    model_name = st.text_input("模型名称", value="deepseek-chat")

# ==========================================
# 3. 文件读取
# ==========================================
uploaded_files = st.file_uploader("第一步：上传评论报表", type=["csv", "xlsx"], accept_multiple_files=True)

all_dfs = []
if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8-sig')  # 默认常用编码
            else:
                df = pd.read_excel(file)
            all_dfs.append(df)
        except:
            st.error(f"{file.name} 读取失败")

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        st.success(f"🎊 数据集成完毕！共计 {len(final_df)} 条评论。")

        # ==========================================
        # 4. 列映射
        # ==========================================
        st.markdown("### 🔍 第二步：智能列映射")
        cols = list(final_df.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_rating = st.selectbox("1. 选择【星级】列", ["无"] + cols)
        with c2:
            sel_title = st.selectbox("2. 选择【标题】列", ["无"] + cols)
        with c3:
            sel_body = st.selectbox("3. 选择【正文】列", cols)

        # ==========================================
        # 5. 可视化分析部分 (Plotly 升级)
        # ==========================================
        st.markdown("---")
        st.markdown("### 📊 第三步：数据可视化概览")

        # 调整列宽比例 [1.5, 1, 1] 放大圆环图的占比
        v_col1, v_col2, v_col3 = st.columns([1.5, 1, 1])

        # 1. Plotly 交互式圆环图
        with v_col1:
            st.markdown("#### ⭐ 星级健康度分布")
            if sel_rating != "无":
                # 准备数据
                rating_data = final_df[sel_rating].value_counts().sort_index(ascending=False).reset_index()
                rating_data.columns = ['Rating', 'Count']
                rating_data['Rating'] = rating_data['Rating'].astype(str) + " Star"

                # 创建 Plotly 圆环图
                fig_plotly = px.pie(
                    rating_data,
                    values='Count',
                    names='Rating',
                    hole=0.5,
                    color_discrete_sequence=['#4A90E2', '#50E3C2', '#F5A623', '#F8E71C', '#D0021B']
                )

                # 调整布局适配暗黑模式
                fig_plotly.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="white"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(t=10, b=10, l=10, r=10)
                )
                fig_plotly.update_traces(textposition='inside', textinfo='percent+label')

                # 在 Streamlit 中显示
                st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.warning("未选择星级列。")


        # 定义生成透明词云图的函数
        def generate_wc(df, col, stops, title_text, col_name, color_scheme):
            text = " ".join(df[col].astype(str).tolist())
            if len(text) > 10:
                wc = WordCloud(stopwords=stops, background_color=None, mode="RGBA",
                               colormap=color_scheme, width=600, height=500).generate(text)
                with col_name:
                    st.markdown(f"#### {title_text}")
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis("off")
                    st.pyplot(fig_wc, transparent=True)


        # 词云图生成
        english_stops = set(stopwords.words('english'))
        extended_stops = STOPWORDS.union(english_stops).union({"product", "amazon", "item", "pajamas", "pajama"})

        if sel_rating != "无" and sel_body != "":
            pos_df = final_df[final_df[sel_rating] >= 4].dropna(subset=[sel_body])
            neg_df = final_df[final_df[sel_rating] <= 3].dropna(subset=[sel_body])
            generate_wc(pos_df, sel_body, extended_stops, "✨ 高频爽点词", v_col2, 'YlGnBu')
            generate_wc(neg_df, sel_body, extended_stops, "⚠️ 高频痛点词", v_col3, 'YlOrRd')

        # ==========================================
        # 6. AI 分析
        # ==========================================
        st.markdown("---")
        if st.button("🔥 第四步：启动 AI 深度洞察分析"):
            if not api_key:
                st.error("请先在侧边栏填入 API Key！")
            else:
                client = OpenAI(api_key=api_key, base_url=base_url)
                sample_df = final_df.dropna(subset=[sel_body]).head(40)

                formatted_reviews = []
                for _, row in sample_df.iterrows():
                    review_parts = []
                    if sel_rating != "无": review_parts.append(f"【{row[sel_rating]}星】")
                    if sel_title != "无": review_parts.append(f"标题: {row[sel_title]}")
                    review_parts.append(f"内容: {row[sel_body]}")
                    full_text = " | ".join(review_parts)
                    formatted_reviews.append(f"- {full_text}")

                all_reviews_text = "\n".join(formatted_reviews)

                prompt = f"""
                你是一名跨境电商产品专家和市场洞察员。请根据以下多维亚马逊评论数据进行深度扫描分析，每一份上传的数据是每个竞品的用户评论数据，请你根据竞品的数据来帮助我自身产品的运营，报告结构需清晰，关键信息加粗：

                1. **【质量红线】**：重点扫描 1-3 星评价，总结最严重的 3 个产品缺陷。
                2. **【用户爽点】**：从 4-5 星评价中提取用户最满意的 3 个卖点。
                3. **【地道关键词建议】**：结合评论标题和正文中的高频表达，提供 5 个高转化的关键词语。
                4. **【产品迭代清单】**：给出下一代产品的具体改进方案。

                数据如下：
                {all_reviews_text}
                """

                with st.spinner('🚀 AI 正在跨维度穿透分析...'):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "你是一个能够精准洞察电商评论深层需求的 AI 专家。"},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7
                        )
                        final_report = response.choices[0].message.content

                        # 使用原生的 success 容器，自带绿色边框和极佳的可读性，彻底解决文字隐身问题
                        st.markdown("### 🏆 AI 深度洞察报告")
                        with st.container():
                            st.info(final_report)

                        st.download_button(
                            label="📥 下载报告为文本文件",
                            data=final_report,
                            file_name="Amazon_VOC_Report.txt",
                            mime="text/plain"
                        )
                    except Exception as ai_err:
                        st.error(f"AI 分析出现错误: {str(ai_err)}")
else:
    st.write("👋 欢迎！请在上方上传亚马逊评论数据开始分析。")
