// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";

import sitemap from "@astrojs/sitemap";

import tailwind from "@astrojs/tailwind";
import { SITE_TITLE, SITE_URL } from "./src/consts";

import wikiLinkPlugin, { defaultUrlResolver } from "@flowershow/remark-wiki-link"
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeMathJaxSvg from "rehype-mathjax/svg";
// import rehypeTypst from "@myriaddreamin/rehype-typst"
import rehypeAsciimath from "@widcardw/rehype-asciimath";
import rehypeCallouts from "rehype-callouts";
import watermarkPlugin from "./watermark-plugin";

import { globSync } from 'node:fs';

// 1. 扫描图片资源 (Public 目录)
const assetDir = 'public/assets/';
const assetFiles = globSync('**/*.{png,jpg,jpeg,gif,webp,svg}', { cwd: assetDir });

const assetPermalinks = assetFiles.reduce((acc, file) => {
  // 保持文件名作为 key，方便 [[image.png]] 匹配
  acc[file] = `/assets/${file}`;
  return acc;
}, /** @type {Record<string, string>} */ ({}));

// 2. 扫描文章资源 (Content 目录)
const postsDir = 'src/content/posts/';
const postFiles = globSync('**/*.{md,mdx}', { cwd: postsDir });

const postPermalinks = postFiles.reduce((acc, file) => {
  // 移除路径和后缀，提取原始文件名作为 key，例如 [[My Post]]
  const fileName = file?.split('/').pop()?.replace(/\.(md|mdx)$/, "") ?? "untitled";

  // 如果 fileName 依然无效（比如是空字符串），提前返回或处理
  if (!fileName || fileName === "untitled") return acc;
  
  // 生成与 Astro 路由一致的 slug
  const slug = fileName
    .toLowerCase()
    .replace(/\s+/g, '-');
    
  acc[fileName] = `/posts/${slug}`; // 假设你的路由是 /posts/[slug]
  return acc;
}, /** @type {Record<string, string>} */ ({}));

// 3. 合并所有永久链接
const permalinks = { ...assetPermalinks, ...postPermalinks };
const files = [...Object.keys(assetPermalinks), ...Object.keys(postPermalinks)];

// console.log(files)

// https://astro.build/config
export default defineConfig({
  site: SITE_URL,
  base: "/",
  integrations: [mdx(), sitemap(), tailwind(), watermarkPlugin(SITE_TITLE)],
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
