import { glob } from 'glob';
import sharp from 'sharp';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';

export default function watermarkPlugin(watermarkText) {
  return {
    name: 'astro-watermark-plugin',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        const distDir = fileURLToPath(dir);
        console.log('🚀 开始半透明、鲁棒性水印处理...');

        const images = await glob('**/*.{jpg,jpeg,png,webp}', {
          cwd: distDir,
          absolute: true,
          nodir: true
        });

        if (images.length === 0) {
          console.warn('⚠️ 未在 dist 目录找到任何图片。');
          return;
        }

        const processed = await Promise.all(images.map(async (imgPath) => {
          try {
            const image = sharp(imgPath);
            const metadata = await image.metadata();

            // 1. 鲁棒性检查：如果读不到尺寸，跳过
            if (!metadata.width || !metadata.height) return false;

            // 避免给极小的图片（如图标）加水印，这里设为 150px
            if (metadata.width < 150) return false;

            // 2. 动态计算水印尺寸 (宽度设为原图的 20%，最小 100px)
            const wWidth = Math.max(Math.floor(metadata.width * 0.20), 100);
            const wHeight = Math.floor(wWidth * 0.3);
            const fontSize = Math.floor(wHeight * 0.4);

            // 3. 生成自适应 SVG (白色文字 + 半透明，适配黑白底)
            const watermarkSvg = Buffer.from(`
              <svg width="${wWidth}" height="${wHeight}">
                <style>
                  .text { 
                    fill: white; 
                    fill-opacity: 0.5; /* 🔥 文字主体 50% 半透明 */
                    font-size: ${fontSize}px; 
                    font-family: sans-serif; 
                    font-weight: bold;
                    paint-order: stroke;
                    stroke: black; /* 黑色描边 */
                    stroke-opacity: 0.2; /* 🔥 描边 20% 半透明，极淡，仅供防白底 */
                    stroke-width: 1px;
                  }
                </style>
                <text x="90%" y="70%" text-anchor="end" class="text">${watermarkText}</text>
              </svg>
            `);

            // 4. 执行合成
            const buffer = await image
              .composite([{
                input: watermarkSvg,
                gravity: 'southeast', // 右下角
                blend: 'over' // 正常叠加
              }])
              .toBuffer();

            await fs.writeFile(imgPath, buffer);
            return true;
          } catch (err) {
            // 针对特定文件报错，不中断整体流程
            console.error(`❌ 跳过文件 (格式不支持或损坏): ${imgPath}`);
            return false;
          }
        }));

        const successCount = processed.filter(Boolean).length;
        console.log(`\n✨ 水印处理完成! 成功: ${successCount} / 总计: ${images.length}`);
      },
    },
  };
}