"use strict";
const gallery = JSON.parse(document.getElementById("gallery-data").textContent);
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const items = gallery.items;
const stateLabels = {detected: "检出候选 · 请复核", target_missed: "目标类别未检出", empty: "未检出任何候选"};
document.getElementById("description").textContent = gallery.description;
const totals = [
  [items.length, "全部图片组"],
  [items.filter(x => x.status === "detected").length, "检出候选的组"],
  [items.filter(x => x.status === "target_missed").length, "目标类未检出的组"],
  [items.filter(x => x.status === "empty").length, "没有任何候选的组"],
];
for (const [count, title] of totals) {
  const box = el("div"); box.append(el("strong", count), el("span", title));
  document.getElementById("summary").append(box);
}
const category = document.getElementById("category");
for (const name of new Set(items.map(x => x.category).filter(Boolean))) {
  const option = el("option", name); option.value = name; category.append(option);
}
const cards = [];
function figure(number, filename, title) {
  const root = el("figure"), stage = el("div", undefined, "image-stage"), link = el("a"), image = el("img");
  link.href = `${number}/${encodeURIComponent(filename)}`;
  image.src = link.href; image.alt = `${number} · ${title}`; image.loading = "lazy";
  link.append(image); stage.append(link); root.append(el("figcaption", title), stage);
  return root;
}
for (const item of items) {
  const root = el("section", undefined, "card"), top = el("div", undefined, "card-top");
  root.id = `group-${item.number}`;
  top.append(el("h2", `第 ${item.number} 组 · ${item.category || "自选图片"} · ${item.source_filename}`),
             el("span", stateLabels[item.status], `badge ${item.status}`));
  const actual = item.detected_classes.length ? item.detected_classes.join("、") : "无";
  const meta = `实际候选：${item.finding_count} 个 · 实际类别：${actual} · 模型：${item.model} · 方案：${item.profile}`;
  const pair = el("div", undefined, "pair");
  pair.append(figure(item.number, item.original, "原始图片"), figure(item.number, "识别结果.jpg", "模型输出与复核状态"));
  root.append(top, el("p", meta, "metadata"), pair); document.getElementById("cards").append(root);
  cards.push({root, item});
}
let status = "";
function filter() {
  const query = document.getElementById("search").value.trim().toLocaleLowerCase();
  let visible = 0;
  for (const {root, item} of cards) {
    const haystack = `${item.number} ${item.category} ${item.source_filename} ${item.detected_classes.join(" ")}`.toLocaleLowerCase();
    const show = (!status || item.status === status) && (!category.value || item.category === category.value) && (!query || haystack.includes(query));
    root.hidden = !show; visible += Number(show);
  }
  document.getElementById("visible-count").textContent = `显示 ${visible} / ${items.length} 组。点击任意图片查看完整尺寸。`;
  document.getElementById("empty-filter").hidden = visible !== 0;
}
document.getElementById("search").addEventListener("input", filter);
category.addEventListener("change", filter);
for (const button of document.querySelectorAll("[data-status]")) {
  button.addEventListener("click", () => {
    status = button.dataset.status;
    for (const candidate of document.querySelectorAll("[data-status]")) candidate.setAttribute("aria-pressed", String(candidate === button));
    filter();
  });
}
filter();
