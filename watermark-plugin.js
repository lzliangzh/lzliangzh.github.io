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
        console.log('🎲 开始随机位置水印处理...');

        const images = await glob('**/*.{jpg,jpeg,png,webp}', {
          cwd: distDir,
          absolute: true,
          nodir: true
        });

        const processed = await Promise.all(images.map(async (imgPath) => {
          try {
            const image = sharp(imgPath);
            const metadata = await image.metadata();

            if (!metadata.width || !metadata.height) return false;
            if (metadata.width < 150) return false;

            // 1. 动态计算水印尺寸 (宽度的 20%)
            const wWidth = Math.max(Math.floor(metadata.width * 0.20), 120);
            const wHeight = Math.floor(wWidth * 0.4);
            const fontSize = Math.floor(wHeight * 0.35);

            // 2. 计算随机位置
            // 预留 5% 的边距 (padding) 避免水印紧贴图片边缘
            const padding = Math.floor(metadata.width * 0.05);
            const maxX = metadata.width - wWidth - padding;
            const maxY = metadata.height - wHeight - padding;

            // 确保坐标不为负数（针对小图）
            const randomLeft = Math.max(padding, Math.floor(Math.random() * maxX));
            const randomTop = Math.max(padding, Math.floor(Math.random() * maxY));

            // 3. 生成 SVG (半透明 + 描边)
            const watermarkSvg = Buffer.from(`
              <svg width="${wWidth}" height="${wHeight}">
                <style>
                  .text { 
                    fill: white; 
                    fill-opacity: 0.5;
                    font-size: ${fontSize}px; 
                    font-family: sans-serif; 
                    font-weight: bold;
                    paint-order: stroke;
                    stroke: black;
                    stroke-opacity: 0.15;
                    stroke-width: 1.5px;
                  }
                </style>
                <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" class="text">
                  ${watermarkText}
                </text>
              </svg>
            `);

            // 4. 执行合成 (不使用 gravity，改用 top/left)
            const buffer = await image
              .composite([{
                input: watermarkSvg,
                top: randomTop,
                left: randomLeft,
                blend: 'over'
              }])
              .toBuffer();

            await fs.writeFile(imgPath, buffer);
            return true;
          } catch (err) {
            console.error(`❌ 跳过文件: ${imgPath}`, err.message);
            return false;
          }
        }));

        const successCount = processed.filter(Boolean).length;
        console.log(`\n✨ 随机水印处理完成! 成功: ${successCount} / 总计: ${images.length}`);
      },
    },
  };
}