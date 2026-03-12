// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

import sitemap from "@astrojs/sitemap";

import tailwind from "@astrojs/tailwind";
import { SITE_URL } from "./src/consts";

import wikiLinkPlugin, { defaultUrlResolver } from "@flowershow/remark-wiki-link"
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeMathJaxSvg from "rehype-mathjax/svg";
// import rehypeTypst from "@myriaddreamin/rehype-typst"
import rehypeAsciimath from "@widcardw/rehype-asciimath";
import rehypeCallouts from "rehype-callouts";

import { file } from "astro/loaders";

import { globSync } from 'glob';

const contentDir = 'src/content/';
const files = globSync('**/*', { cwd: contentDir });
const permalinks = files.reduce((acc, file) => {
  // 1. 检查是否为图片格式
  const isImage = /\.(png|jpg|jpeg|gif|webp|svg)$/i.test(file);
  
  if (isImage) {
    // 如果是图片，指向资源目录（根据你实际存放位置修改，例如 /public/src/ 或 /src/assets/）
    acc[file] = `/src/${file}`;
  } else {
    // 2. 如果是 Markdown (md/mdx)，处理成文章路由
    // 移除后缀名以便匹配 [[filename]]
    const slug = file
      .replace(/\.(md|mdx)$/, "") 
      .toLowerCase()
      .replace(/\s+/g, '-'); // 确保这里的逻辑和你的 [...slug].astro 一致
      
    acc[file] = `/${slug}`;
  }
  
  return acc;
}, /** @type {Record<string, string>} */ ({}));

// console.log(files)

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  base: "/",
  integrations: [mdx(), sitemap(), tailwind()],
  markdown: {
    shikiConfig: {
      themes: {
        light: "catppuccin-latte",
        dark: "catppuccin-mocha",
      },
    },
    remarkPlugins: [
      remarkMath,
      [wikiLinkPlugin, {
        files: files,
        format: 'shortestPossible',
        permalinks: permalinks,
      }]
    ],
    rehypePlugins: [
      rehypeAsciimath,
      // [rehypeKatex, {output: "html"}],
      rehypeMathJaxSvg,
      rehypeCallouts,
    ],
  },
});
