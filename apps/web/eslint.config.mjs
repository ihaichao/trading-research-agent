// eslint-config-next 16 直接导出 flat config，不再需要 FlatCompat
import coreWebVitals from "eslint-config-next/core-web-vitals";
import typescript from "eslint-config-next/typescript";

const config = [
  {
    ignores: [".next/**", "node_modules/**", "public/**", "src/types/report.ts"],
  },
  ...coreWebVitals,
  ...typescript,
];

export default config;
