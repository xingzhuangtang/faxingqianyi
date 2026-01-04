import hashlib
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼大模型素描转换模块
使用阿里云通义万相图生图API实现高质量素描效果
"""

import os
import time
import requests
from http import HTTPStatus
import dashscope
from dashscope import ImageSynthesis


class BailianSketchConverter:
    """百炼素描转换器"""
    
    def get_deterministic_seed(self, image_path):
        """基于图片内容生成确定性种子，确保相同输入产生相同输出"""
        try:
            with open(image_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            seed = int(file_hash[:8], 16) % 10000
            return seed
        except:
            return 42

    def __init__(self, api_key=None):
        """
        初始化
        
        Args:
            api_key: 百炼API Key,如果不提供则从环境变量DASHSCOPE_API_KEY读取
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("未找到DASHSCOPE_API_KEY,请设置环境变量或传入api_key参数")
        
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        # 素描风格prompt模板 - 已重命名
        self.style_prompts = {
            'pencil': '将这张照片转换为铅笔素描风格,保持人物五官特征完全清晰,细腻的线条,柔和的阴影',
            'anime': 'Japanese anime style VIBRANT COLORED illustration, clean precise linework with RICH SATURATED ANIME COLORS, meticulously detailed hair with CLEARLY SEPARATED COLORFUL STRANDS, each hair strand showing DISTINCT SPATIAL DEPTH and VIVID COLOR LAYERING, PRONOUNCED spatial sense with colorful foreground and background hair layers, rich saturated anime colors with professional cel-shading technique, exquisite facial features with typical anime aesthetics and SOFT SKIN TONES, large expressive eyes with detailed highlights COLORFUL iris and VIBRANT reflections, elaborate hair texture with DIMENSIONAL COLORED LAYERS clearly visible, professional anime art style with clean BLACK outlines, VIVID MULTI-COLOR PALETTE typical of Japanese animation with BRIGHT HUES, CLEAR DEPTH PERCEPTION with overlapping COLORED hair strands, masterful anime style with SPATIAL HIERARCHY in COLORFUL hair rendering, each hair layer at DIFFERENT DEPTH PLANES with DISTINCT COLORS creating strong effect, COLORFUL SHADING and HIGHLIGHTS throughout the portrait',
            'ink': 'Traditional Chinese SUBTLE COLORED ink wash painting with LIGHT COLOR SATURATION at 30 percent, delicate brushwork with GENTLE COLORED ink strokes in soft muted hues, elegant hair rendering with VISIBLE LAYERED STRANDS showing DEPTH and SPATIAL SEPARATION, each hair layer clearly DISTINCT with spatial sense between layers, SOFT PASTEL COLOR GRADATION with restrained color palette, artistic interpretation with refined LIGHT COLORED strokes, masterful ink wash technique showing hair VOLUME DEPTH and DIMENSIONAL LAYERS, refined facial features with delicate PALE COLORED ink lines, expressive eyes with precise ink detailing, professional Sumi-e style with MUTED SUBTLE COLORS, dynamic hair strokes with natural LIGHT COLORED ink gradation, CLEAR SPATIAL RELATIONSHIPS between hair strands, PRONOUNCED LAYERING EFFECT with foreground middle and background hair clearly separated, GENTLE COLOR TONES throughout',
            'vivid': 'Vibrant colored sketch style with 10 to 30 percent COLOR SATURATION, pencil sketch foundation with SUBTLE COLOR ACCENTS, maintaining clear sketch lines with LIGHT PASTEL COLOR TOUCHES, preserving character features with GENTLE COLOR HINTS, artistic beauty with RESTRAINED COLORFUL ELEMENTS, soft color wash over detailed pencil work, MUTED COLOR PALETTE with delicate hues, sketch texture visible through LIGHT COLOR LAYERS, balanced monochrome and SUBTLE COLOR combination'
        }
    
    def convert(self, image_url, style='ink', watermark=False):
        """
        将图像转换为素描风格
        
        Args:
            image_url: 图像URL(支持公网URL、Base64、本地文件路径)
            style: 素描风格,可选值: pencil, anime, ink, vivid
            watermark: 是否添加水印
        
        Returns:
            tuple: (素描图像URL, 处理信息dict)
        """
        print(f"\n🎨 开始百炼素描转换...")
        print(f"   风格: {style}")
        print(f"   输入: {image_url[:100]}...")
        
        start_time = time.time()
        
        try:
            prompt = self.style_prompts.get(style, self.style_prompts['ink'])
            
            print(f"   📤 调用通义万相API...")
            rsp = ImageSynthesis.call(
                api_key=self.api_key,
                model="wan2.5-i2i-preview",
                prompt=prompt,
                images=[image_url],
                negative_prompt="低分辨率,模糊,失真,变形,五官改变",
                n=1,
                watermark=watermark
            )
            
            if rsp.status_code != HTTPStatus.OK:
                error_msg = f"API调用失败: {rsp.code} - {rsp.message}"
                print(f"   ❌ {error_msg}")
                return None, {'success': False, 'error': error_msg}
            
            result_url = rsp.output.results[0].url
            elapsed = time.time() - start_time
            
            print(f"   ✅ 素描转换成功!")
            print(f"   耗时: {elapsed:.2f}秒")
            print(f"   结果URL: {result_url[:100]}...")
            
            info = {
                'success': True,
                'style': style,
                'elapsed_time': f"{elapsed:.2f}秒",
                'result_url': result_url,
                'task_id': rsp.output.task_id,
                'prompt': prompt
            }
            
            return result_url, info
            
        except Exception as e:
            error_msg = f"素描转换异常: {str(e)}"
            print(f"   ❌ {error_msg}")
            return None, {'success': False, 'error': error_msg}
    
    def download_result(self, result_url, save_path):
        """
        下载素描结果图像
        
        Args:
            result_url: 结果图像URL
            save_path: 保存路径
        
        Returns:
            bool: 是否成功
        """
        try:
            print(f"\n📥 下载素描结果...")
            print(f"   URL: {result_url[:100]}...")
            print(f"   保存到: {save_path}")
            
            response = requests.get(result_url, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"   ✅ 下载成功!")
            return True
            
        except Exception as e:
            print(f"   ❌ 下载失败: {str(e)}")
            return False


def test_converter():
    """测试素描转换器"""
    print("=" * 60)
    print("百炼素描转换器测试")
    print("=" * 60)
    
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 未设置DASHSCOPE_API_KEY环境变量")
        return
    
    print(f"✅ API Key已设置: {api_key[:20]}...")
    
    try:
        converter = BailianSketchConverter(api_key)
        print("✅ 素描转换器创建成功")
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return


if __name__ == '__main__':
    test_converter()
