// Loaded in <head>: the theme is applied before first paint (no flash)
const store = {
  get: (key) => { try { return localStorage.getItem(key); } catch { return null; } },
  set: (key, value) => { try { localStorage.setItem(key, value); } catch {} },
};

document.documentElement.dataset.theme = store.get("theme") || "light";

addEventListener("DOMContentLoaded", () => {
  document.getElementById("theme").onclick = () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    store.set("theme", theme);
  };
});
