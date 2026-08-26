import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 报告数据目前来自 public/samples 下的静态 JSON（由 scripts/sync-fixtures.mjs
  // 从 docs/samples 同步）。M6 接入后端后改为从 API 读取。
  reactStrictMode: true,
};

export default nextConfig;
