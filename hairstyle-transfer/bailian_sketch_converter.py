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
    
    def __init__(self, api_key=None):
        """
        初始化
        
        Args:
            api_key: 百炼API Key,如果不提供则从环境变量DASHSCOPE_API_KEY读取
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("未找到DASHSCOPE_API_KEY,请设置环境变量或传入api_key参数")
        
        # 设置API端点(北京地域)
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        
        # 素描风格prompt模板
        self.style_prompts = {
            'pencil': '将这张照片转换为铅笔素描风格,保持人物五官特征完全清晰,细腻的线条,柔和的阴影',
            'detailed': '细节丰富的素描画,强调轮廓和阴影,保持人物特征一致,专业素描技法',
            'artistic': '艺术素描风格,黑白线条,对比强烈,保持人物面部特征清晰,高级艺术感',
            'colored': '彩色素描风格,保留适当颜色,素描线条明显,保持人物特征不变,艺术美感'
        }
    
    def convert(self, image_url, style='artistic', watermark=False):
        """
        将图像转换为素描风格
        
        Args:
            image_url: 图像URL(支持公网URL、Base64、本地文件路径)
            style: 素描风格,可选值: pencil, detailed, artistic, colored
            watermark: 是否添加水印
        
        Returns:
            tuple: (素描图像URL, 处理信息dict)
        """
        print(f"\n🎨 开始百炼素描转换...")
        print(f"   风格: {style}")
        print(f"   输入: {image_url[:100]}...")
        
        start_time = time.time()
        
        try:
            # 获取prompt
            prompt = self.style_prompts.get(style, self.style_prompts['artistic'])
            
            # 调用通义万相API
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
            
            # 检查响应
            if rsp.status_code != HTTPStatus.OK:
                error_msg = f"API调用失败: {rsp.code} - {rsp.message}"
                print(f"   ❌ {error_msg}")
                return None, {'success': False, 'error': error_msg}
            
            # 获取结果URL
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
    
    # 检查API Key
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ 未设置DASHSCOPE_API_KEY环境变量")
        print("   请执行: export DASHSCOPE_API_KEY='your-api-key'")
        return
    
    print(f"✅ API Key已设置: {api_key[:20]}...")
    
    # 创建转换器
    try:
        converter = BailianSketchConverter(api_key)
        print("✅ 素描转换器创建成功")
    except Exception as e:
        print(f"❌ 创建失败: {e}")
        return
    
    # 测试图像URL
    test_url = "https://img.alicdn.com/imgextra/i2/O1CN01FuGdH91RenU9KPeri_!!6000000002137-2-tps-1344-896.png"
    
    # 测试各种风格
    styles = ['pencil', 'detailed', 'artistic', 'colored']
    
    for style in styles:
        print(f"\n{'=' * 60}")
        print(f"测试风格: {style}")
        print(f"{'=' * 60}")
        
        result_url, info = converter.convert(test_url, style=style)
        
        if info['success']:
            print(f"\n✅ {style}风格转换成功!")
            print(f"   耗时: {info['elapsed_time']}")
            print(f"   结果URL: {result_url[:100]}...")
            print(f"   任务ID: {info['task_id']}")
        else:
            print(f"\n❌ {style}风格转换失败!")
            print(f"   错误: {info['error']}")
        
        # 等待一下避免频率限制
        time.sleep(2)
    
    print(f"\n{'=' * 60}")
    print("测试完成")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    test_converter()
