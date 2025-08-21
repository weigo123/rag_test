"""
测试文件，用于测试大模型接口
"""
from dotenv import load_dotenv
import os
import openai
from google import genai
from pathlib import Path
def encode_image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read())
        decode_data = base64_data.decode("utf-8")
        return decode_data

# VLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
VLM_MODEL = "glm-4.5v"
load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")
base_url = os.getenv("ZHIPU_BASE_URL")

def test_iamge_translation()-> None:
    # api_key = os.getenv("LOCAL_API_KEY")  
    # base_url = os.getenv("LOCAL_BASE_URL")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    image_path = "./test_data/test.jpg"
    image_base64 = encode_image_to_base64(image_path)

    filename = "联邦制药-港股公司研究报告-创新突破三靶点战略联姻诺和诺德-25071225页.pdf"
    system_prompt = "你是一个专业的股票分析师，可以熟练分析股票行业、股票财务报表和研究报告"
    user_prompt = f"这是一张有关上市公司的图片，请详细描述这张图片的信息，图片所属文件名称为《${filename}》"
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],   
        max_tokens=1024,
        temperature=0.1
    )
    desc = response.choices[0].message.content
    print(desc)

def test_gemini_text_embedding()->None:
    client = genai.Client()

def test_table_translation(filename:str, image_path:str, parse_text:str)-> None:
    image = Path(image_path)
    if not image.exists():
        raise ValueError(f"Image {image_path} does not exist.")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    image_base64 = encode_image_to_base64(image_path=image_path)
    system_prompt = "你是一个专业的股票分析师，可以熟练分析股票行业、股票财务报表和研究报告"
    # user_prompt = f"""这是一张有关上市公司的表格的图片，请详细描述这张图片的信息，图片所属文件名称为<{filename}>
    #     分析以下方面:
    #         1. 主要内容/主题
    #         2. 包含的关键信息点

    #         输出格式必须严格为:
    #         {{
    #         "title": "标题100字以内",
    #         "description": "详细描述1000字以内"
    #         }}
    #         只返回JSON，不要有其他说明文字，不能有尖括号类似的文字。
    # """
    user_prompt = f"""
        这是一张有关上市公司的表格的图片，请详细描述这张图片的信息，图片来自上市公司的研究报告，图片所属文件名称为<{filename}>
        详细描述图片中的关键信息点
        """
    response = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": f"{system_prompt}"},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1024,
        temperature=0.2
    )
    desc = response.choices[0].message.content
    print(desc)
    
if __name__ == "__main__":
    filename = "艾力斯-公司深度报告商业化成绩显著产品矩阵持续拓宽-25070718页"
    image_path = "./test_data/images/e615166fd385400608ed9069eb20088bb78ce2c5de7fad66da7072570597386f.jpg"
    test_table_translation(filename=filename, image_path=image_path, parse_text="")


