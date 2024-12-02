import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import time
import logging  
from pathlib import Path
import joblib
import os
from pathlib import Path
import streamlit as st
import base64

# 设置页面标题  
st.markdown("<h1 style='text-align: center;'>PART1 基于饮食习惯的个性化肥胖程度分析🥗</h1>", unsafe_allow_html=True) 

# 侧边栏成功提示
st.sidebar.success("请完成数据上传后再使用侧边栏功能喔❤")


# 获取脚本所在目录的绝对路径
path = str(Path(__file__).parent.absolute())

# 构建视频文件的绝对路径
video_file_path = Path(path) / "fit.mp4"

# 打开视频文件
video_file = open(video_file_path, 'rb')
video_bytes = video_file.read()

st.markdown(
    """
    ### 一、意义
    随着生活方式的改变，肥胖问题日益严重。了解和掌握影响肥胖的因素，并对肥胖程度进行分类和预测，对于肥胖的预防和控制策略的制定具有重要意义。本项目旨在通过数据分析，为个性化肥胖预防和干预提供依据，帮助人们更好地管理健康。
"""
)


st.write("# 介绍🗠")
#本地视频
st.video(video_bytes,format="mp4",start_time=2)
# 添加出处链接  
st.markdown("[出处链接](https://www.bilibili.com/video/BV1s54y1r7Ed/?spm_id_from=333.337.search-card.all.click&vd_source=096a17588c803a5145918c456427c022)")  

# 页面主体内容
st.markdown(
    """
    ### 二、特征
    我们的数据集中包含了多个与饮食习惯和身体状况相关的特征，如性别、年龄、身高、体重、家族肥胖史、经常吃高热量食物、日常蔬菜摄入、每日主餐次数、餐间进食、吸烟情况、饮水量、每日卡路里监测情况、身体活动频率、使用电子设备时间、饮酒频率、日常交通方式等。这些特征与肥胖程度有着不同程度的相关性，例如家族肥胖史、常吃高热量食物等与肥胖程度呈明显正相关，而性别、吸烟情况等相关性较弱。
"""
)
import streamlit as st  
import pandas as pd  

# 创建数据  
data = {  
    "变量名称": [  
        "Gender（性别）",  
        "Age（年龄）",  
        "Height（身高）",  
        "Weight（体重）",  
        "family_history_with_overweight（家族肥胖史）",  
        "FAVC（经常吃高热量食物）",  
        "FCVC（日常蔬菜摄入）",  
        "NCP（每日主餐次数）",  
        "CAEC（餐间进食）",  
        "SMOKE（吸烟情况）",  
        "Water（饮水量）",  
        "CALC（家族肥胖史）",  
        "FAVC（经常吃高热量食物）",  
        "NObeyesdad（日常蔬菜摄入）"  
    ],  
    "作用": [  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征",  
        "特征"  
    ],  
    "类型": [  
        "分类变量",  
        "连续变量",  
        "连续变量",  
        "连续变量",  
        "二分类变量",  
        "二分类变量",  
        "整数变量",  
        "连续变量",  
        "分类变量",  
        "二分类变量",  
        "分类变量",  
        "二分类变量",  
        "二分类变量",  
        "整数变量"  
    ],  
    "人口统计学信息": [  
        "性别",  
        "年龄",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无"  
    ],  
    "描述": [  
        "无",  
        "无",  
        "无",  
        "无",  
        "家族成员是否曾患或正患有超重？",  
        "你是否经常吃高热量食物？",  
        "你在餐食中通常是否吃蔬菜？",  
        "你每天有几顿主餐？",  
        "你在两餐之间是否吃任何食物？",  
        "你吸烟吗？",  
        "你每天喝多少水？",  
        "家族成员是否曾患或正患有超重？",  
        "你是否经常吃高热量食物？",  
        "你在餐食中通常是否吃蔬菜？"  
    ],  
    "单位": [  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无"  
    ],  
    "缺失值情况": [  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无",  
        "无"  
    ]  
}  

# 将数据转换为 DataFrame  
df = pd.DataFrame(data)  


# 显示数据表  
st.dataframe(df)

st.markdown(
    """
    ### 三、任务
      本项目的主要任务是利用机器学习方法，基于个人的饮食习惯和身体状况数据，准确预测肥胖程度。我们通过数据预处理、模型选择与优化等一系列步骤，构建了有效的预测模型。用户可以输入相关数据，通过在网页中的**交互**，获取**个性化**的肥胖程度预测结果，从而更好地了解自身健康状况，采取相应的预防和干预措施。
    """
)







# 全局变量，用于存储数据集和相关模型等信息
df = None
encoder = None
columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
best_model = None
import base64


# 用户上传数据集函数
def upload_dataset():
    global df, encoder, X, y, X_train, X_test, y_train, y_test, best_model
    st.subheader("→请先上传数据集哟！")
    uploaded_file = st.file_uploader("请使用您的数据集文件（.CSV格式）如果您没有专门的数据集，可以下载并使用上面我们为您准备的测试用例数据集）", type="csv")

    # 添加模型上传功能
    uploaded_model_file = st.file_uploader("请上传模型文件（.pkl格式）如果您没有专门的数据集，可以下载并使用下面我们为您准备的测试模型", type="pkl")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # 数据预处理
        encoder = LabelEncoder()
        for column in columns_to_encode:
            df[column] = encoder.fit_transform(df[column])

        # 划分特征和目标变量
        X = df.drop('NObeyesdad', axis=1)
        y = df['NObeyesdad']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if uploaded_model_file is not None:
            # 获取脚本所在目录的绝对路径
            path = str(Path(__file__).parent.absolute())

            # 构建临时模型文件存储路径
            model_temp_path = Path(path) / "temp_model.pkl"

            # 将上传的模型文件保存到临时路径
            with open(model_temp_path, 'wb') as f:
                f.write(uploaded_model_file.read())

            # 加载用户上传的模型
            best_model = joblib.load(model_temp_path)

            # 删除临时文件
            os.remove(model_temp_path)

            st.sidebar.info(f"✅ 数据已上传！已成功加载您上传的模型！请使用侧边栏进行个性化数据配置。")
        else:
            st.warning("请上传模型文件以便进行后续操作(请使用您训练的模型，或者使用我们训练好的XGBoost模型)。")

    # 设置xgb_model.pkl下载按钮及绝对路径
    # 获取脚本所在目录的绝对路径
    path = str(Path(__file__).parent.absolute())

    # 构建xgb_model.pkl的绝对路径
    xgb_model_path = Path(path) / "xgb_model.pkl"

    if os.path.exists(xgb_model_path):
        with open(xgb_model_path, 'rb') as f:
            xgb_model_bytes = f.read()
        st.download_button(
            label="下载xgb_model.pkl（测试模型）",
            data=xgb_model_bytes,
            file_name="xgb_model.pkl",
            mime="application/octet-stream"
        )

def get_user_input_sidebar():
    st.sidebar.subheader("请输入以下个人信息：")

    gender = st.sidebar.selectbox("性别", ["女", "男"], index=1)
    gender_value = 1 if gender == "男" else 0

    age = st.sidebar.number_input("年龄", min_value=1, step=1)
    height = st.sidebar.number_input("身高（厘米）", min_value=1, step=1)
    weight = st.sidebar.number_input("体重（公斤）", min_value=1, step=1)

    family_history = st.sidebar.selectbox("家族是否有肥胖史", ["否", "是"], index=0)
    family_history_value = 1 if family_history == "是" else 0

    favc = st.sidebar.selectbox("是否经常吃高热量食物", ["否", "是"], index=0)
    favc_value = 1 if favc == "是" else 0

    fcvc = st.sidebar.selectbox("餐食中通常是否吃蔬菜", ["否", "是"], index=1)
    fcvc_value = 1 if fcvc == "是" else 0

    ncp = st.sidebar.number_input("每天吃多少顿正餐", min_value=1, step=1)

    caec = st.sidebar.selectbox("两餐之间是否吃食物", ["否", "是"], index=0)
    caec_value = 1 if caec == "是" else 0

    smoke = st.sidebar.selectbox("是否吸烟", ["否", "是"], index=0)
    smoke_value = 1 if smoke == "是" else 0

    ch2o = st.sidebar.number_input("每天喝多少水（毫升）", min_value=1, step=1)

    scc = st.sidebar.selectbox("是否监控每天摄入的卡路里", ["否", "是"], index=0)
    scc_value = 1 if scc == "是" else 0

    faf = st.sidebar.selectbox("进行体育活动的频率", ["很少", "偶尔", "经常"], index=1)
    faf_value = {"很少": 0, "偶尔": 1, "经常": 2}[faf]

    tue = st.sidebar.number_input("每天使用科技设备的时间（小时）", min_value=0, step=1)

    calc = st.sidebar.selectbox("家族成员是否患有超重或超重", ["否", "是"], index=0)
    calc_value = 1 if calc == "是" else 0

    mtrans = st.sidebar.selectbox("交通方式", ["步行", "骑自行车", "开车", "公共交通"], index=0)
    mtrans_value = {"步行": 0, "骑自行车": 1, "开车": 2, "公共交通": 3}[mtrans]

    user_data = {
        'Gender': gender_value,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history_value,
        'FAVC': favc_value,
        'FCVC': fcvc_value,
        'NCP': ncp,
        'CAEC': caec_value,
        'SMOKE': smoke_value,
        'CH2O': ch2o,
        'SCC': scc_value,
        'FAF': faf_value,
        'TUE': tue,
        'CALC': calc_value,
        'MTRANS': mtrans_value
    }

    if st.sidebar.button("提交信息并获取建议"):
        return pd.DataFrame(user_data, index=[0])
    else:
        return None

# 检查数据函数
def check_data(user_data):
    global df
    # 检查列名是否与训练数据一致
    expected_columns = df.columns.tolist()[:-1]  # 排除目标变量列
    if set(user_data.columns)!= set(expected_columns):
        st.error("输入数据的列名与训练数据不匹配。")
        return False

    # 检查分类变量的值是否在合理范围内
    for column in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
        if user_data[column].iloc[0] not in [0, 1]:
            st.error(f"{column} 的值必须为0或1。")
            return False

    # 检查FCVC的值是否在合理范围内（假设为0或1）
    if user_data['FCVC'].iloc[0] not in [0, 1]:
        st.error("FCVC 的值必须为0或1。")
        return False

    # 检查体育活动频率（FAF）的值是否在合理范围内
    if user_data['FAF'].iloc[0] not in [0, 1, 2]:
        st.error("FAF 的值必须为0、1或2。")
        return False

    # 检查交通方式（MTRANS）的值是否在合理范围内
    if user_data['MTRANS'].iloc[0] not in [0, 1, 2, 3]:
        st.error("MTRANS 的值必须为0、1, 2或3。")
        return False

    return True

# 预测用户肥胖程度
def predict_obesity(user_data):
    global encoder
    user_data_encoded = user_data.copy()
    for column in columns_to_encode[:-1]:
        # 先对encoder进行拟合
        encoder.fit(df[column])
        user_data_encoded[column] = encoder.transform(user_data_encoded[column])
    prediction = best_model.predict(user_data_encoded)
    return prediction[0]

# 给出建议
def give_suggestions(prediction, user_data):
    if prediction == 0:
        st.success("根据您提供的信息，您可能体重不足。以下是一些针对性建议：")
        if user_data['Age'].iloc[0] < 18:
            st.write("您正处于生长发育阶段，要确保摄入足够的蛋白质、钙等营养物质，比如多喝牛奶、吃鸡蛋、鱼肉等，以促进身体正常发育。")
        else:
            st.write("建议您增加营养摄入，保证每日三餐规律进食，适当增加富含优质蛋白的食物，如瘦肉、豆类等，以及碳水化合物的摄取量，可选择全麦面包、糙米等健康的碳水来源。")
    elif prediction == 1:
        st.success("根据您提供的信息，您的体重处于正常范围。继续保持健康的饮食习惯和适量的运动，有助于维持良好的身体状态。以下是一些保持建议：")
        if user_data['FAF'].iloc[0] == 0:
            st.write("您目前体育活动频率较低，可适当增加一些简单的运动，比如每天散步30分钟或者每周进行两次瑜伽练习，以进一步提升身体素质。")
        elif user_data['FAF'].iloc[0] == 1:
            st.write("您的运动频率尚可，继续保持目前的运动习惯，同时注意饮食的均衡搭配，多吃蔬菜水果，少吃油腻和高糖食物。")
        elif user_data['FAF'].iloc[0] == 2:
            st.write("您经常进行体育活动，非常棒！记得要合理安排休息时间，保证充足的睡眠，让身体有足够的时间恢复，同时也要注意饮食营养的全面性。")
    elif prediction == 2:
        st.success("根据您提供的信息，您处于超重I级。以下是一些有助于改善的建议：")
        if user_data['FAVC'].iloc[0] == 1:
            st.write("您经常吃高热量食物，这可能是导致超重的原因之一。建议您控制此类食物的摄入，比如减少油炸食品、甜品的食用频率，可多吃一些蔬菜沙拉、水果作为替代。")
        if user_data['SMOKE'].iloc[0] == 1:
            st.write("吸烟对健康有诸多不利影响，也可能与体重超标有关。考虑戒烟或者逐渐减少吸烟量，这不仅有助于控制体重，还能改善整体健康状况。")
        st.write("增加蔬菜和水果的摄取，每周至少进行三次中等强度的运动，如快走、慢跑等，每次运动时间建议在30分钟以上。")
    elif prediction == 3:
        st.success("根据您提供的 信息，您处于超重II级。需要更加严格控制饮食，减少糖分和油脂的摄入，增加运动量，可考虑加入一些力量训练来提高基础代谢率。以下是具体建议：")
        if user_data['CH2O'].iloc[0] < 1500:
            st.write("您每天的饮水量可能不足，充足的水分摄入有助于新陈代谢，建议您每天至少喝1500 - 2000毫升水，以帮助身体排出多余的废物和毒素。")
        if user_data['SCC'].iloc[0] == 0:
            st.write("您目前没有监控每天摄入的卡路里，建议您开始关注饮食的热量摄入，可使用一些饮食记录APP来帮助您了解自己每天吃了多少热量，以便更好地控制饮食。")
        st.write("增加运动量，除了有氧运动外，可每周进行2 - 3次力量训练，如深蹲、平板支撑等，每次训练时间可根据自身情况安排在20 - 30分钟左右。")
    elif prediction == 4:
        st.success("根据您提供的信息，您属于肥胖I型。以下是一些针对性的建议：")
        if user_data['CALC'].iloc[0] == 1:
            st.write("您家族成员有超重或肥胖情况，遗传因素可能在您的体重问题上起到一定作用。但通过健康的生活方式调整仍然可以改善。建议您咨询专业的营养师制定个性化的饮食计划，同时增加运动量，如每天进行至少30分钟的有氧运动，如游泳、跳绳等。")
        if user_data['MTRANS'].iloc[0] == 2 or user_data['MTRANS'].iloc[0] == 3:
            st.write("您的交通方式可能相对比较久坐，比如开车或乘坐公共交通。尽量增加步行或骑自行车的机会，比如提前一两站下车步行去目的地，或者在短距离出行时选择骑自行车，以增加日常活动量。")
        st.write("饮食方面，要严格控制高热量、高脂肪、高糖食物的摄入，增加蔬菜、水果和高纤维食物的摄取量。")
    elif prediction == 5:
        st.success("根据您提供的信息，您属于肥胖II型。强烈建议您尽快寻求医生或专业健康顾问的帮助，制定全面的减肥计划，包括饮食调整、运动方案以及可能的医疗干预。以下是一些初步建议：")
        if user_data['FCVC'].iloc[0] == 0:
            st.write("您餐食中通常不吃蔬菜，蔬菜对于维持身体健康和控制体重非常重要。请务必增加蔬菜的摄入量，每餐都应保证有一定量的蔬菜，可做成蔬菜汤、清炒蔬菜等多种形式。")
        if user_data['NCP'].iloc[0] > 3:
            st.write("您每天吃的正餐顿数较多，可能导致热量摄入过多。考虑适当减少正餐顿数，或者控制每餐的食量，同时保证饮食的营养均衡。")
        st.write("运动方面，要逐步增加运动量和运动强度，在专业人员的指导下进行系统的训练，包括有氧运动和力量训练相结合。")
    elif prediction == 6:
        st.success("根据您提供的信息，您属于肥胖III型。这是非常严重的肥胖情况，请立即就医，在专业人员的指导下进行系统的治疗和生活方式的改变。以下是一些紧急建议：")
        if user_data['family_history_with_overweight'].iloc[0] == 1:
            st.write("您家族有肥胖史，遗传因素加上您目前的体重状况，情况较为严峻。请严格按照医生的建议进行治疗和生活方式调整，包括饮食控制、运动安排以及可能的药物治疗等。")
        if user_data['CAEC'].iloc[0] == 1:
            st.write("您两餐之间还吃食物，这可能进一步增加了热量摄入。请尽量避免两餐之间进食，如需进食，可选择低热量、高纤维的食物，如水果、坚果（少量）等。")
        st.write("要全方位地改变生活方式，包括严格的饮食管理、规律的运动锻炼以及定期的健康检查等。")

# 找出所有相关指标及其重要性程度并输出函数
def output_all_related_indicators(user_data):
    global best_model, X
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    importance_dict = dict(zip(feature_names, feature_importances))

    st.subheader("**三、基于您填写的信息，各相关指标及其对体重状况的分析**")

    # 分析性别对体重的可能影响
    gender = "男" if user_data['Gender'].iloc[0] == 1 else "女"
    if user_data['Gender'].iloc[0] == 1:
        st.markdown("您的性别为 <span style='color: lightblue; font-weight: bold;'>男</span>👨，一般来说男性相对女性可能更容易堆积肌肉，基础代谢率可能稍高，但这也取决于其他生活习惯因素，如运动和饮食等。结合您目前填写的其他信息来看，您的体重情况可能受到多种因素综合影响。", unsafe_allow_html=True)
    else:
        st.markdown("您的性别为 <span style='color: lightpink; font-weight: bold;'>女</span>👩，女性身体成分相对男性可能有更多脂肪比例，不过通过合理的饮食和运动习惯也能很好地维持健康体重。就您目前填写的各项信息而言，整体在影响着您的体重状况。", unsafe_allow_html=True)

    # 分析年龄对体重的可能影响
    age = user_data['Age'].iloc[0]
    if age < 18:
        st.markdown("您处于 <span style='color: green; font-weight: bold;'>成长发育时期</span>🧒，身体还在生长发育阶段，这个阶段的体重变化可能更多与身体发育需求相关，合理的营养摄入对健康成长尤为重要，目前您的体重情况也会受此阶段特点及其他生活习惯共同影响。", unsafe_allow_html=True)
    elif age < 30:
        st.markdown("您处于 <span style='color: orange; font-weight: bold;'>青年时期</span>👱，新陈代谢相对较快，一般较容易维持体重，但如果饮食不均衡或缺乏运动，也可能出现体重波动，结合您填写的信息，比如您的饮食和运动习惯等都在对当前体重产生作用。", unsafe_allow_html=True)
    elif age < 50:
        st.markdown("您处于 <span style='color: purple; font-weight: bold;'>中年时期</span>👨‍🦱，新陈代谢可能开始逐渐变慢，需要更加关注饮食和运动以维持体重稳定，从您提供的信息看，像您的饮食结构（如是否经常吃高热量食物、餐食中是否吃蔬菜等）以及体育活动频率等都与当前体重状况息息相关。", unsafe_allow_html=True)
    else:
        st.markdown("您处于 <span style='color: gray; font-weight: bold;'>老年时期</span>👴，身体机能有所下降，新陈代谢更慢，体重管理可能需要更精细的饮食控制和适度的运动，您填写的各项生活习惯信息都在影响着目前的体重情况。", unsafe_allow_html=True)

    # 分析身高与体重的关系及影响
    height = user_data['Height'].iloc[0]
    weight = user_data['Weight'].iloc[0]
    bmi = weight / ((height / 100) ** 2)
    st.markdown(f"根据您填写的身高({height}厘米)和体重({weight}公斤)计算得出的BMI值为 <span style='color: red; font-weight: bold;'>{bmi:.2f}</span>。一般来说，BMI在不同范围对应不同的体重状况类别，您可参考相关标准进一步了解自己的体重情况。结合其他填写信息，如饮食、运动等习惯也在持续影响着这个数值以及您的实际体重状况。", unsafe_allow_html=True)

    # 分析家族肥胖史对体重的可能影响
    if user_data['family_history_with_overweight'].iloc[0] == 1:
        st.markdown("您家族有肥胖史😟，这可能意味着您在基因层面存在一定的肥胖易感性，但并不意味着一定会肥胖，通过健康的生活方式，如合理饮食、规律运动等可以有效降低肥胖风险，就目前您填写的各项信息来看，这些生活习惯正在与遗传因素共同作用于您的体重状况。", unsafe_allow_html=True)
    else:
        st.markdown("您家族没有肥胖史🎉，这在一种程度上是个优势，但仍需保持良好的生活习惯以维持健康体重，从您填写的信息可知，您当前的饮食、运动等习惯都在决定着您的体重是否能持续保持在健康范围。", unsafe_allow_html=True)

    # 分析饮食相关指标对体重的可能影响
    if user_data['FAVC'].iloc[0] == 1:
        st.markdown("您经常吃高热量食物🍔，这是导致体重增加的一个重要因素，需要注意控制此类食物的摄入，结合您填写的其他饮食相关信息，如餐食中是否吃蔬菜、是否监控每天摄入的卡路里等，都在影响着您的整体饮食热量摄入情况从而影响体重。", unsafe_allow_html=True)
    else:
        st.markdown("您不经常吃高热量食物🥗，这对维持体重是有帮助的，再配合您填写的其他饮食方面信息，比如餐食中吃蔬菜的情况、每天正餐顿数等，共同塑造了您目前的饮食热量摄入模式，进而影响体重。", unsafe_allow_html=True)

    if user_data['FCVC'].iloc[0] == 1:
        st.markdown("您餐食中通常吃蔬菜🥦，这是非常好的饮食习惯，蔬菜富含膳食纤维等营养物质，有助于增加饱腹感、促进肠道蠕动，对控制体重有积极作用，结合其他饮食相关信息如是否经常吃高热量食物等，共同影响着您的体重状况。", unsafe_allow_html=True)
    else:
        st.markdown("您餐食中通常不吃蔬菜😕，蔬菜在饮食中占有重要地位，缺乏蔬菜摄入可能导致营养不均衡、膳食纤维不足等问题，进而影响体重控制，结合您填写的其他饮食相关信息如每天正餐顿数等，都在对体重产生影响。", unsafe_allow_html=True)

    ncp = user_data['NCP'].iloc[0]
    if ncp > 3:
        st.markdown("您每天吃的正餐顿数较多🍽️，这可能会导致热量摄入过多，需要注意合理控制每餐的食量以及食物种类，结合您填写的其他饮食相关信息如是否经常吃高热量食物、是否监控每天摄入的卡路里等，来综合管理饮食热量摄入，从而影响体重。", unsafe_allow_html=True)
    else:
        st.markdown("您每天吃的正餐顿数相对合理👍，再配合您填写的其他饮食相关信息如餐食中是否吃蔬菜等，有助于维持健康的饮食热量摄入模式，进而影响体重。", unsafe_allow_html=True)

    if user_data['CAEC'].iloc[0] == 1:
        st.markdown("您两餐之间还吃食物🍪，这可能会增加额外的热量摄入，不利于体重控制，结合您填写的其他饮食相关信息如是否经常吃高热量食物、是否监控每天摄入的卡路里等，需要注意合理选择两餐之间的食物，以避免体重增加。", unsafe_allow_html=True)
    else:
        st.markdown("您两餐之间不吃食物🥳，这有助于减少不必要的热量摄入，对维持体重有帮助，结合您填写的其他饮食相关信息如餐食中是否吃蔬菜等，共同塑造了您目前的饮食热量管理情况，进而影响体重。", unsafe_allow_html=True)

    if user_data['SCC'].iloc[0] == 1:
        st.markdown("您监控每天摄入的卡路里📈，这是非常好的体重管理习惯，能够让您清楚了解自己的热量摄入情况，结合您填写的其他饮食相关信息如是否经常吃高热量食物、餐食中是否吃蔬菜等，有助于您更精准地控制饮食，从而影响体重。", unsafe_allow_html=True)
    else:
        st.markdown("您没有监控每天摄入的卡路里😕，可能会导致对自己的热量摄入情况不太清楚，结合您填写的其他饮食相关信息如是否经常吃高热量食物、餐食中是否吃蔬菜等，建议您可以考虑开始监控卡路里摄入，以便更好地管理体重。", unsafe_allow_html=True)

    # 分析运动相关指标对体重的可能影响
    if user_data['SMOKE'].iloc[0] == 1:
        st.markdown("您吸烟🚬，吸烟不仅对身体健康有诸多危害，还可能影响新陈代谢，进而影响体重，结合您填写的其他信息如体育活动频率等，建议您考虑戒烟，以改善整体健康状况和体重管理。", unsafe_allow_html=True)
    else:
        st.markdown("您不吸烟🎉，这对身体健康和体重管理是有好处的，再结合您填写的体育活动频率等信息，有助于维持健康的生活方式和体重状况。", unsafe_allow_html=True)

    if user_data['FAF'].iloc[0] == 0:
        st.markdown("您进行体育活动的频率很低😕，这不利于维持体重和身体健康，建议您增加体育活动的频率和强度，结合您填写的其他信息如每天使用科技设备的时间等，合理安排时间进行运动，以改善体重状况。", unsafe_allow_html=True)
    elif user_data['FAF'].iloc[0] == 1:
        st.markdown("您进行体育活动的频率为偶尔👍，这是一个还可以的情况，但可以进一步提高体育活动的频率和强度，结合您填写的其他信息如每天使用科技设备的时间等，更好地平衡生活和运动，以维持健康的体重状况。", unsafe_allow_html=True)
    elif user_data['FAF'].iloc[0] == 2:
        st.markdown("您进行体育活动的频率较高👏，这非常好，继续保持并合理安排运动强度和休息时间，结合您填写的其他信息如每天使用科技设备的时间等，有助于维持良好的体重状况和身体健康。", unsafe_allow_html=True)

    # 分析交通方式对体重的作用
    if user_data['MTRANS'].iloc[0] == 0:
        st.markdown("您选择步行🚶作为交通方式，这是一种很好的增加日常活动量的方式，对体重控制和身体健康都有积极作用，结合您填写的其他信息如体育活动频率等，有助于维持健康的体重状况。", unsafe_allow_html=True)
    elif user_data['MTRANS'].iloc[0] == 1:
        st.markdown("您选择骑自行车🚲作为交通方式，这也是一种不错的增加日常活动量的方式，对体重控制和身体健康有帮助，结合您填写的其他信息如体育活动频率等，有助于维持健康的体重状况。", unsafe_allow_html=True)
    elif user_data['MTRANS'].iloc[0] == 2:
        st.markdown("您选择开车🚗作为交通方式，相对来说开车时活动量较少，需要注意在其他方面增加活动量，比如增加步行上下车的距离、停车后多走动等，结合您填写的其他信息如体育活动频率等，以维持健康的体重状况。", unsafe_allow_html=True)
    elif user_data['MTRANS'].iloc[0] == 3:
        st.markdown("您选择公共交通🚌作为交通方式，虽然在乘坐过程中活动量也有限，但可以利用上下车等机会增加活动量，结合您填写的其他信息如体育活动频率等，有助于维持健康的体重状况。", unsafe_allow_html=True)

    # 以表格形式展示部分重要信息及分析
    st.markdown("### 部分重要信息及分析汇总：")

    # 提取相关信息用于表格展示
    gender = "男" if user_data['Gender'].iloc[0] == 1 else "女"
    age = user_data['Age'].iloc[0]
    bmi = user_data['Weight'].iloc[0] / ((user_data['Height'].iloc[0] / 100) ** 2)
    family_history = "有" if user_data['family_history_with_overweight'].iloc[0] == 1 else "无"
    high_calorie_food = "是" if user_data['FAVC'].iloc[0] == 1 else "否"
    vegetable_intake = "是" if user_data['FCVC'].iloc[0] == 1 else "否"
    meal_count = f"{user_data['NCP'].iloc[0]}顿"
    between_meal_eating = "是" if user_data['CAEC'].iloc[0] == 1 else "否"
    calorie_monitoring = "是" if user_data['SCC'].iloc[0] == 1 else "否"
    physical_activity_frequency = f"{user_data['FAF'].iloc[0]}（0-很低，1-偶尔，2-较高）"
    transportation_mode = "步行" if user_data['MTRANS'].iloc[0] == 0 else "骑自行车" if user_data['MTRANS'].iloc[0] == 1 else "开车" if user_data['MTRANS'].iloc[0] == 2 else "公共交通"

    # 分析各项对体重的影响并生成对应描述
    gender_weight_impact = ""
    if user_data['Gender'].iloc[0] == 1:
        gender_weight_impact = "男性相对易堆积肌肉，可能因基础代谢率稍高影响体重，但受运动、饮食等多种因素综合作用。"
    else:
        gender_weight_impact = "女性身体成分相对男性可能脂肪比例稍高，通过合理饮食和运动可维持健康体重，受多种因素影响。"

    age_weight_impact = ""
    if age < 18:
        age_weight_impact = "处于成长发育时期，身体生长发育需求影响体重，需保证营养摄入，受此阶段特点及其他生活习惯共同影响。"
    elif age < 30:
        age_weight_impact = "处于青年时期，新陈代谢快，饮食不均衡或缺乏运动可能致体重波动，受饮食、运动习惯等影响。"
    elif age < 50:
        age_weight_impact = "处于中年时期，新陈代谢逐渐变慢，饮食结构和体育活动频率等与体重状况息息相关。"
    else:
        age_weight_impact = "处于老年时期，身体机能下降、新陈代谢更慢，需精细饮食控制和适度运动来管理体重，受生活习惯影响。"

    bmi_weight_impact = f"BMI值为{bmi:.2f}，不同范围对应不同体重状况，结合饮食、运动等习惯影响实际体重状况。"

    family_history_weight_impact = ""
    if user_data['family_history_with_overweight'].iloc[0] == 1:
        family_history_weight_impact = "家族有肥胖史，存在肥胖易感性，通过健康生活方式可降低风险，生活习惯与遗传因素共同作用于体重。"
    else:
        family_history_weight_impact = "家族无肥胖史，是维持健康体重的优势，当前饮食、运动等习惯决定体重是否保持健康范围。"

    high_calorie_food_weight_impact = ""
    if user_data['FAVC'].iloc[0] == 1:
        high_calorie_food_weight_impact = "经常吃高热量食物，是导致体重增加因素，需控制摄入，结合其他饮食信息影响整体热量摄入及体重。"
    else:
        high_calorie_food_weight_impact = "不常吃高热量食物，有助于维持体重，配合其他饮食方面信息塑造当前热量摄入模式影响体重。"

    vegetable_intake_weight_impact = ""
    if user_data['FCVC'].iloc[0] == 1:
        vegetable_intake_weight_impact = "餐食中常吃蔬菜，富含膳食纤维等营养，有助于增加饱腹感、促进肠道蠕动，利于控制体重，结合其他饮食信息影响体重状况。"
    else:
        vegetable_intake_weight_impact = "餐食中通常不吃蔬菜，可能导致营养不均衡、膳食纤维不足，影响体重控制，结合其他饮食信息对体重有影响。"

    meal_count_weight_impact = ""
    if user_data['NCP'].iloc[0] > 3:
        meal_count_weight_impact = "每天正餐顿数较多，可能导致热量摄入过多，需合理控制食量及食物种类，结合其他饮食信息管理热量摄入影响体重。"
    else:
        meal_count_weight_impact = "每天正餐顿数相对合理，配合其他饮食信息有助于维持健康的热量摄入模式，进而影响体重。"

    between_meal_eating_weight_impact = ""
    if user_data['CAEC'].iloc[0] == 1:
        between_meal_eating_weight_impact = "两餐之间还吃食物，可能增加额外热量摄入，不利于体重控制，需合理选择食物，结合其他饮食信息避免体重增加。"
    else:
        between_meal_eating_weight_impact = "两餐之间不吃食物，有助于减少不必要热量摄入，对维持体重有帮助，结合其他饮食信息塑造热量管理情况影响体重。"

    calorie_monitoring_weight_impact = ""
    if user_data['SCC'].iloc[0] == 1:
        calorie_monitoring_weight_impact = "监控每天摄入的卡路里，能清楚了解热量摄入情况，有助于更精准地控制饮食，结合其他饮食信息影响体重。"
    else:
        calorie_monitoring_weight_impact = "未监控每天摄入的卡路里，可能对热量摄入情况不清楚，建议考虑监控以更好地管理体重，结合其他饮食信息。"

    physical_activity_frequency_weight_impact = ""
    if user_data['FAF'].iloc[0] == 0:
        physical_activity_frequency_weight_impact = "体育活动频率很低，不利于维持体重和身体健康，建议增加频率和强度，结合其他信息合理安排运动改善体重状况。"
    elif user_data['FAF'].iloc[0] == 1:
        physical_activity_frequency_weight_impact = "体育活动频率为偶尔，可进一步提高频率和强度，结合其他信息更好地平衡生活和运动以维持健康体重状况。"
    elif user_data['FAF'].iloc[0] == 2:
        physical_activity_frequency_weight_impact = "体育活动频率较高，非常好，继续保持并合理安排运动强度和休息时间，结合其他信息维持良好体重状况和身体健康。"

    transportation_mode_weight_impact = ""
    if user_data['MTRANS'].iloc[0] == 0:
        transportation_mode_weight_impact = "选择步行作为交通方式，是增加日常活动量的好方式，对体重控制和身体健康有积极作用，结合体育活动频率等信息维持健康体重状况。"
    elif user_data['MTRANS'].iloc[0] == 1:
        transportation_mode_weight_impact = "选择骑自行车作为交通方式，也是增加日常活动量的不错方式，对体重控制和身体健康有帮助，结合体育活动频率等信息维持健康体重状况。"
    elif user_data['MTRANS'].iloc[0] == 2:
        transportation_mode_weight_impact = "选择开车作为交通方式，开车时活动量较少，需在其他方面增加活动量，结合体育活动频率等信息维持健康体重状况。"
    else:
        transportation_mode_weight_impact = "选择公共交通作为交通方式，虽乘坐过程中活动量有限，但可利用上下车等机会增加活动量，结合体育活动频率等信息维持健康体重状况。"

    table_data = {
        "指标": ["性别", "年龄", "BMI", "家族肥胖史", "高热量食物摄入", "餐食蔬菜摄入", "正餐顿数", "两餐间进食", "监控卡路里", "体育活动频率", "交通方式"],
        "情况": [gender, f"{age}岁", f"{bmi:.2f}", family_history, high_calorie_food, vegetable_intake, meal_count, between_meal_eating, calorie_monitoring, physical_activity_frequency, transportation_mode],
        "对体重影响分析": [gender_weight_impact, age_weight_impact, bmi_weight_impact, family_history_weight_impact, high_calorie_food_weight_impact, vegetable_intake_weight_impact, meal_count_weight_impact, between_meal_eating_weight_impact, calorie_monitoring_weight_impact, physical_activity_frequency_weight_impact, transportation_mode_weight_impact]
    }
    styled_table = pd.DataFrame(table_data)
    styled_table = styled_table.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#f2f2f2'), ('text-align', 'center'), ('font-size', '18px')]},
        {'selector': 'td', 'props': [('text-align', 'center'), ('font-size', '16px')]}
    ])
    # 设置表格高度和宽度，使其显示更大
    styled_table = styled_table.set_properties(**{'height': '500px', 'width': '100%'})

    # 直接使用st.dataframe展示带有样式的表格
    st.dataframe(styled_table)
    # 给出健康计划相关网址链接
    st.markdown("如需制定更详细的健康计划，您可以参考以下网址：[中国营养学会官网](https://www.cnsoc.org/)")

# 新增函数：展示并提供测试用例数据集下载
def show_and_download_test_dataset():
    st.subheader("**测试用例数据集**")
    # 获取脚本所在目录的绝对路径
    path = str(Path(__file__).parent.absolute())

    # 构建测试数据文件的绝对路径
    test_file_path = Path(path) / "fixed_encoded_dataset.csv"

    # 读取测试数据
    test_df = pd.read_csv(test_file_path)

    # 展示测试数据集
    st.dataframe(test_df)

    # 提供下载按钮
    csv = test_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下载测试用例数据集",
        data=csv,
        file_name="test_encoded_dataset.csv",
        mime="text/csv"
    )

# 在main函数中调用该函数，在上传数据集之前展示测试用例数据集
def main():
    st.markdown("<h1 style='text-align: center;'>PART2 基于XGBoost和特征学习的个性化肥胖程度分析🥦</h1>", unsafe_allow_html=True)
    show_and_download_test_dataset()

    upload_dataset()

    user_data = get_user_input_sidebar()
    if user_data is not None:
        if check_data(user_data):
            prediction = predict_obesity(user_data)
            give_suggestions(prediction, user_data)
            output_all_related_indicators(user_data)
        else:
            st.error("数据检查未通过，请检查输入数据。")

if __name__ == "__main__":
    main()

