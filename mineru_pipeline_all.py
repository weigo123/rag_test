
import os
from pathlib import Path
import json
from collections import defaultdict
import asyncio
from typing import Optional
from image_utils.async_image_analysis import AsyncImageAnalysis
from image_utils.prompts import get_image_analysis_prompt
def parse_all_pdfs(datas_dir, output_base_dir):
    """
    步骤1：解析所有PDF，输出内容到 data_base_json_content/
    """
    from mineru_parse_pdf import do_parse
    datas_dir = Path(datas_dir)
    output_base_dir = Path(output_base_dir)
    pdf_files = list(datas_dir.rglob('*.pdf'))
    if not pdf_files:
        print(f"未找到PDF文件于: {datas_dir}")
        return
    for pdf_path in pdf_files:
        file_name = pdf_path.stem
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        output_dir = output_base_dir / file_name
        if output_dir.exists():
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["ch"],
            backend="vlm-sglang-engine",
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True
        )
        print(f"已输出: {output_dir / 'auto' / (file_name + '_content_list.json')}")

def group_by_page(content_list):
    pages = defaultdict(list)
    for item in content_list:
        page_idx = item.get('page_idx', 0)
        pages[page_idx].append(item)
    return dict(pages)

def item_to_markdown(item, enable_image_caption=True, file_name="", file_dir:Optional[Path]=None):
    """
    enable_image_caption: 是否启用多模态视觉分析（图片caption补全），默认True。
    """
    # 默认API参数：硅基流动Qwen/Qwen2.5-VL-32B-Instruct
    # vision_provider = "guiji"
    # vision_model = "Qwen/Qwen2.5-VL-32B-Instruct"
    # vision_api_key = os.getenv("GUIJI_API_KEY")
    # vision_base_url = os.getenv("GUIJI_BASE_URL")
    vision_provider = "zhipu"
    vision_model = "ZhipuAI/glm-4.5v"
    vision_api_key = os.getenv("ZHIPU_API_KEY")
    vision_base_url = os.getenv("ZHIPU_BASE_URL")
                               
    if item['type'] == 'text':
        level = item.get('text_level', 0)
        text = item.get('text', '')
        if level == 1:
            return f"# {text}\n\n"
        elif level == 2:
            return f"## {text}\n\n"
        else:
            return f"{text}\n\n"
    elif item['type'] == 'image':
        captions = item.get('image_caption', [])
        caption = captions[0] if captions else ''
        img_path = item.get('img_path', '')
        print(f"正在处理图片: file_dir={file_dir},image_path={img_path}")
        img_path = str((file_dir / img_path).resolve())
        print(f"enable_image_caption={enable_image_caption},caption={caption},img_path={img_path},exists={os.path.exists(img_path)}")
        # 如果没有caption，且允许视觉分析，调用多模态API补全
        if enable_image_caption and not caption and img_path and os.path.exists(img_path):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async def get_caption():
                    async with AsyncImageAnalysis(
                        provider=vision_provider,
                        api_key=vision_api_key,
                        base_url=vision_base_url,
                        vision_model=vision_model,
                        max_concurrent=8
                    ) as analyzer:
                        role_prompt = "你是一个专业的股票分析师"
                        ability_prompt = "可以熟练分析股票行业、股票财务报表和研究报告"
                        prompt = get_image_analysis_prompt(title_max_length=30, 
                                                           description_max_length=1000,
                                                           role_prompt=role_prompt,
                                                           ability_prompt=ability_prompt,
                                                           file_name=file_name
                                                           )
                        print(prompt)
                        result = await analyzer.analyze_image(local_image_path=img_path, prompt=prompt)
                        print(result)
                        return result.get('title') or result.get('description') or ''
                caption = loop.run_until_complete(get_caption())
                loop.close()
                if caption:
                    item['image_caption'] = [caption]
            except Exception as e:
                print(f"图片解释失败: {img_path}, {e}")
        md = f"![{caption}]({img_path})\n"
        return md + "\n"
    elif item['type'] == 'table':
        captions = item.get('table_caption', [])
        caption = captions[0] if captions else ''
        table_html = item.get('table_body', '')
        img_path = item.get('img_path', '')
        md = ''
        if caption:
            md += f"**{caption}**\n"
        if img_path:
            md += f"![{caption}]({img_path})\n"
        md += f"{table_html}\n\n"
        return md
    else:
        return '\n'

def assemble_pages_to_markdown(pages, file_name:str, file_dir: Path):
    """
    将按页面分组的内容列表转换为Markdown格式文本
    
    参数:
        pages (dict): 按页面索引分组的内容字典，键为页码，值为该页所有内容项的列表
                     例如: {0: [item1, item2, ...], 1: [item3, item4, ...]}
    
    返回:
        dict: 键为页码，值为该页完整Markdown文本的字典
              例如: {0: "# 标题\n\n正文内容\n\n![图片描述](图片路径)\n\n", 
                    1: "## 章节标题\n\n表格内容\n\n"}
    
    处理流程:
        1. 遍历每一页的内容项
        2. 对每个内容项调用 item_to_markdown 函数转换为Markdown格式
        3. 将同一页的所有Markdown内容拼接成完整的页面文本
        4. 返回所有页面的Markdown文本字典
    """
    page_md = {}
    for page_idx in sorted(pages.keys()):
        md = ''
        for item in pages[page_idx]:
            md += item_to_markdown(item, enable_image_caption=True, file_name=file_name, file_dir=file_dir)
        page_md[page_idx] = md
    return page_md

def process_all_pdfs_to_page_json(input_base_dir, output_base_dir):
    """
    步骤2：将 content_list.json 转为 page_content.json
    """
    input_base_dir = Path(input_base_dir)
    output_base_dir = Path(output_base_dir)
    pdf_dirs = [d for d in input_base_dir.iterdir() if d.is_dir()]
    for pdf_dir in pdf_dirs:
        file_name = pdf_dir.name
        json_path = pdf_dir / 'auto' / f'{file_name}_content_list.json'
        if not json_path.exists():
            sub_dir = pdf_dir / file_name
            json_path2 = sub_dir / 'auto' / f'{file_name}_content_list.json'
            if json_path2.exists():
                json_path = json_path2
            else:
                print(f"未找到: {json_path} 也未找到: {json_path2}")
                continue
        with open(json_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)
        pages = group_by_page(content_list)
        print(f"开始处理assemble_pages_to_markdown")
        page_md = assemble_pages_to_markdown(pages, file_name, file_dir=json_path.parent)
        output_dir = output_base_dir / file_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path = output_dir / f'{file_name}_page_content.json'
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(page_md, f, ensure_ascii=False, indent=2)
        print(f"已输出: {output_json_path}")

def process_page_content_to_chunks(input_base_dir, output_json_path):
    """
    步骤3：将 page_content.json 合并为 all_pdf_page_chunks.json
    
    Args:
        input_base_dir (str or Path): 包含PDF处理结果的根目录路径，该目录下应包含各个PDF文件对应的子目录
        output_json_path (str or Path): 输出合并后JSON文件的完整路径
        
    Returns:
        None: 函数无返回值，结果直接写入到output_json_path指定的文件中
        
    功能说明:
        1. 遍历input_base_dir目录下的所有子目录
        2. 在每个子目录中查找page_content.json文件
        3. 读取每个page_content.json文件的内容
        4. 将所有内容重新组织为统一格式的chunks
        5. 将所有chunks写入到一个JSON文件中
    """
    input_base_dir = Path(input_base_dir)
    all_chunks = []
    for pdf_dir in input_base_dir.iterdir():
        if not pdf_dir.is_dir():
            continue
        file_name = pdf_dir.name
        page_content_path = pdf_dir / f"{file_name}_page_content.json"
        if not page_content_path.exists():
            sub_dir = pdf_dir / file_name
            page_content_path2 = sub_dir / f"{file_name}_page_content.json"
            if page_content_path2.exists():
                page_content_path = page_content_path2
            else:
                print(f"未找到: {page_content_path} 也未找到: {page_content_path2}")
                continue
        with open(page_content_path, 'r', encoding='utf-8') as f:
            page_dict = json.load(f)
        for page_idx, content in page_dict.items():
            chunk = {
                "id": f"{file_name}_page_{page_idx}",
                "content": content,
                "metadata": {
                    "page": page_idx,
                    "filename": file_name + ".pdf"
                }
            }
            all_chunks.append(chunk)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"已输出: {output_json_path}")

def main():
    base_dir = Path(__file__).parent
    datas_dir = base_dir / 'datas'
    content_dir = base_dir / 'data_base_json_content'
    page_dir = base_dir / 'data_base_json_page_content'
    chunk_json_path = base_dir / 'all_pdf_page_chunks.json'
    # 步骤1：PDF → content_list.json
    parse_all_pdfs(datas_dir, content_dir)
    # 步骤2：content_list.json → page_content.json
    process_all_pdfs_to_page_json(content_dir, page_dir)
    # 步骤3：page_content.json → all_pdf_page_chunks.json
    process_page_content_to_chunks(page_dir, chunk_json_path)
    print("全部处理完成！")

if __name__ == '__main__':
    main()
