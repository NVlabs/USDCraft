// Navigation enhancement; the page and video work without JavaScript.
const links = [...document.querySelectorAll('.nav-links a')];
const sections = links.map(link => document.querySelector(link.getAttribute('href')));
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      for (const link of links) {
        const active = link.hash === `#${entry.target.id}`;
        link.classList.toggle('active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      }
    }
  }, { rootMargin: '-15% 0px -55% 0px', threshold: 0 });
  sections.forEach(section => observer.observe(section));
}

// Each application stage keeps all supplied clips available without loading them at once.
document.querySelectorAll('.clip-select').forEach(select => {
  select.addEventListener('change', () => {
    const video = select.closest('.application-card').querySelector('video');
    const option = select.selectedOptions[0];
    video.pause();
    video.muted = true;
    video.poster = option.dataset.poster;
    video.setAttribute('aria-label', option.textContent);
    video.querySelector('source').src = option.value;
    video.querySelector('a').href = option.value;
    video.load();
  });
});
// Keep audio and motion from competing when another demo starts.
document.querySelectorAll('video').forEach(video => {
  // Only the demo (data-audio) keeps its sound; every other clip is silent.
  if (!video.hasAttribute('data-audio')) {
    video.defaultMuted = true;
    video.muted = true;
  }
  video.addEventListener('play', () => {
    // Looping showcase clips (data-ambient) play alongside each other.
    if (video.classList.contains('opening-video') || video.hasAttribute('data-ambient')) return;
    document.querySelectorAll('video').forEach(other => {
      if (other !== video && !other.paused) other.pause();
    });
  });
});

// Opening motion is silent, pausable, and only runs while visible.
const openingVideo = document.querySelector('.opening-video');
const openingToggle = document.querySelector('.opening-toggle');
if (openingVideo && openingToggle) {
  let userPaused = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let visible = true;
  function syncLabel() {
    const paused = openingVideo.paused;
    openingToggle.textContent = paused ? '▶' : 'Ⅱ';
    openingToggle.setAttribute('aria-label', paused ? 'Play opening animation' : 'Pause opening animation');
  }
  function syncPlayback() {
    const otherPlaying = [...document.querySelectorAll('video')].some(v => v !== openingVideo && !v.hasAttribute('data-ambient') && !v.paused);
    if (visible && !document.hidden && !userPaused && !otherPlaying) openingVideo.play().catch(syncLabel);
    else openingVideo.pause();
    syncLabel();
  }
  openingToggle.addEventListener('click', () => { userPaused = !openingVideo.paused; syncPlayback(); });
  openingVideo.addEventListener('play', syncLabel);
  openingVideo.addEventListener('pause', syncLabel);
  new IntersectionObserver(entries => { visible = entries[0].isIntersecting; syncPlayback(); }, {threshold:0.15}).observe(openingVideo);
  document.addEventListener('visibilitychange', syncPlayback);
  syncPlayback();
}

const cinema = document.querySelector('.opening');
if (cinema && openingVideo) {
  const reveal = () => cinema.classList.add('title-visible');
  openingVideo.addEventListener('timeupdate', () => { if (openingVideo.currentTime >= 5.7) reveal(); });
  // Reveal on scroll or keyboard navigation as well as after the opening shot.
  window.addEventListener('scroll', () => { if (scrollY > 90) reveal(); }, {passive:true});
  cinema.addEventListener('focusin', reveal);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) reveal();
  setTimeout(reveal, 8500);
}
