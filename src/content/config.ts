import { defineCollection, z } from "astro:content";

const posts = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    description: z.string().default(""),
    date: z.coerce.date(),
    image: z.string().default(""),
    link: z.string().default(""),
    tags: z.array(z.string()).default([]),
    category: z.string().default("")
  }),
});

export const collections = { posts };
