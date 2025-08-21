\
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片分析相关的提示词模板
"""
from typing import Optional
def get_image_analysis_prompt(title_max_length: int, description_max_length: int, 
                              role_prompt:Optional[str]="",
                              ability_prompt:Optional[str]="",
                              file_name:Optional[str]="") -> str:
    """
    生成图片分析的提示词。

    Args:
        title_max_length: 标题最大长度。
        description_max_length: 描述最大长度。

    Returns:
        格式化的提示词字符串。
    """
    return f"""{role_prompt},{ability_prompt},图片来自{file_name},分析这张图片并生成一个{title_max_length}字以上的标题、{description_max_length}字以上的图片描述，使用JSON格式输出。

分析以下方面:
1. 图像类型（图表、示意图、照片等）
2. 主要内容/主题
3. 包含的关键信息点

输出格式必须严格为:
{{
  "title": "标题({title_max_length}字以内)",
  "description": "详细描述({description_max_length}字以内)"
}}

只返回JSON，不要有其他说明文字。
"""


def get_table_analysis_prompt(title_max_length: int, description_max_length: int, 
                              role_prompt:Optional[str]="",
                              ability_prompt:Optional[str]="",
                              file_name:Optional[str]="") -> str:
    """
    生成图片分析的提示词。

    Args:
        title_max_length: 标题最大长度。
        description_max_length: 描述最大长度。

    Returns:
        格式化的提示词字符串。
    """
    return f"""{role_prompt},{ability_prompt},图片来自{file_name},这张图片来自一张表格，分析表格并给出{title_max_length}字以上的标题、{description_max_length}字以上的图片描述，使用JSON格式输出。

分析以下方面:
2. 主要内容/主题
3. 包含的信息点

输出格式必须严格为:
{{
  "title": "标题({title_max_length}字以内)",
  "description": "详细描述({description_max_length}字以内)"
}}

只返回JSON，不要有其他说明文字。
"""
