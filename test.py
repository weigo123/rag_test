from dotenv import load_dotenv
import os
import openai
def encode_image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as f:
        base64_data = base64.b64encode(f.read())
        decode_data = base64_data.decode("utf-8")
        return decode_data

# VLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
VLM_MODEL = "glm-4.5v"
if __name__ == "__main__":
    load_dotenv()
    # api_key = os.getenv("LOCAL_API_KEY")  
    # base_url = os.getenv("LOCAL_BASE_URL")
    api_key = os.getenv("ZHIPU_API_KEY")
    base_url = os.getenv("ZHIPU_BASE_URL")
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


