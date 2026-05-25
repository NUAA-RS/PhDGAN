# 测试代码 test.py
import torch
from torchvision import transforms
from PIL import Image
import os
from generator_gam_deep import EnhancedGenerator

# 参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G1 = EnhancedGenerator().to(device)
G2 = EnhancedGenerator().to(device)
G1.load_state_dict(torch.load('checkpoints/G1_epoch100.pth', map_location=device))
G2.load_state_dict(torch.load('checkpoints/G2_epoch100.pth', map_location=device))
G1.eval()
G2.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 测试函数
# 修改后的测试函数
def process_image(input_path, output_path, generator, polar_type):
    img = Image.open(input_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = generator(img_tensor).squeeze().cpu().numpy()

    # 转换并保存图像（保持原始文件名）
    output = (output.transpose(1, 2, 0) * 0.5 + 0.5) * 255
    output_img = Image.fromarray(output.astype('uint8'))
    output_img.save(output_path)  # 直接使用传入的完整路径保存

# 执行测试
# 执行测试（修改后）
if __name__ == '__main__':
    # 输入输出路径设置
    input_base_dir = 'dataset_color/train_data1/test'  # 测试数据根目录
    output_base_dir = 'xxx'  # 结果保存根目录

    # 创建输出目录结构
    os.makedirs(os.path.join(output_base_dir, 'VH'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'VV'), exist_ok=True)

    # 批量处理VH极化图像
    vh_dir = os.path.join(input_base_dir, 'VH')
    for filename in os.listdir(vh_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 处理单张图像
            input_path = os.path.join(vh_dir, filename)
            output_path = os.path.join(output_base_dir, 'VH', filename)
            process_image(input_path, output_path, G1, 'VH')
            print(f'Processed VH: {filename}')

    # 批量处理VV极化图像
    vv_dir = os.path.join(input_base_dir, 'VV')
    for filename in os.listdir(vv_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(vv_dir, filename)
            output_path = os.path.join(output_base_dir, 'VV', filename)
            process_image(input_path, output_path, G2, 'VV')
            print(f'Processed VV: {filename}')

    print("Batch processing completed!")