// 把 packages/contracts/samples 下的样例报告复制到 public/samples，供前端在没有后端时开发。
// 由 predev / prebuild 自动执行，不需要手动跑。
import { cp, mkdir, readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "..", "..", "packages", "contracts", "samples");
const dest = join(here, "..", "public", "samples");

await mkdir(dest, { recursive: true });

let files;
try {
  files = await readdir(src);
} catch {
  console.warn(`[sync-fixtures] ${src} 不存在，跳过。先在仓库根目录跑 \`make schema\`。`);
  process.exit(0);
}

const jsons = files.filter((f) => f.endsWith(".json"));
for (const file of jsons) {
  await cp(join(src, file), join(dest, file));
}
console.log(`[sync-fixtures] copied ${jsons.length} fixture(s) -> public/samples`);
