// Small progressive enhancement for the owner shell. Core routing remains in
// the existing bounded browser adapter; this only controls presentation.
document.addEventListener('DOMContentLoaded', () => {
  const nav = document.querySelector('.product-nav');
  if (!nav) return;
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'nav-more';
  toggle.textContent = 'More views';
  toggle.setAttribute('aria-expanded', 'false');
  toggle.addEventListener('click', () => {
    const expanded = nav.classList.toggle('show-advanced');
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.textContent = expanded ? 'Fewer views' : 'More views';
  });
  nav.append(toggle);
});
