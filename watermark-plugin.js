import { glob } from 'glob';
import sharp from 'sharp';
import fs from 'fs/promises';
import { fileURLToPath } from 'url';
import path from 'path';

export default function watermarkPlugin(watermarkText) {
  return {
    name: 'astro-watermark-plugin',
    hooks: {
      'astro:build:done': async ({ dir }) => {
        const distDir = fileURLToPath(dir);
        console.log('📦 开始随机水印并执行图片压缩...');

        const images = await glob('**/*.{jpg,jpeg,png,webp}', {
          cwd: distDir,
          absolute: true,
          nodir: true
        });

        const processed = await Promise.all(images.map(async (imgPath) => {
          try {
            const ext = path.extname(imgPath).toLowerCase();
            const image = sharp(imgPath);
            const metadata = await image.metadata();

            if (!metadata.width || !metadata.height) return false;

            // 1. 水印逻辑 (保持之前的随机逻辑)
            const wWidth = Math.max(Math.floor(metadata.width * 0.20), 120);
            const wHeight = Math.floor(wWidth * 0.4);
            const fontSize = Math.floor(wHeight * 0.35);

            const padding = Math.floor(metadata.width * 0.05);
            const maxX = metadata.width - wWidth - padding;
            const maxY = metadata.height - wHeight - padding;
            const randomLeft = Math.max(padding, Math.floor(Math.random() * maxX));
            const randomTop = Math.max(padding, Math.floor(Math.random() * maxY));

            const watermarkSvg = Buffer.from(`
              <svg width="${wWidth}" height="${wHeight}">
                <style>
                  .text { fill: white; fill-opacity: 0.5; font-size: ${fontSize}px; font-family: sans-serif; font-weight: bold; paint-order: stroke; stroke: black; stroke-opacity: 0.15; stroke-width: 1.5px; }
                </style>
                <text x="50%" y="50%" text-anchor="middle" dominant-baseline="middle" class="text">${watermarkText}</text>
              </svg>
            `);

            // 2. 压缩逻辑：根据格式应用不同的优化
            let pipeline = image.composite([{
              input: watermarkSvg,
              top: randomTop,
              left: randomLeft,
              blend: 'over'
            }]);

            if (ext === '.jpg' || ext === '.jpeg') {
              pipeline = pipeline.jpeg({ quality: 60, progressive: true, mozjpeg: true });
            } else if (ext === '.png') {
              // png 压缩：使用 palette 减色模式能大幅缩小体积
              pipeline = pipeline.png({ quality: 60, palette: true });
            } else if (ext === '.webp') {
              pipeline = pipeline.webp({ quality: 80 });
            }

            const buffer = await pipeline.toBuffer();
            await fs.writeFile(imgPath, buffer);
            return true;
          } catch (err) {
            console.error(`❌ 处理失败: ${imgPath}`, err.message);
            return false;
          }
        }));

        const successCount = processed.filter(Boolean).length;
        console.log(`\n✨ 完成！已压缩并加水印: ${successCount} 张图片`);
      },
    },
  };
}