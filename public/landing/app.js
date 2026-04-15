// LIVE TIME
// select the element liveTime from the Document Object Model(DOM) 
function updateTime() {
  document.getElementById("liveTime").innerText =
    "Company Time: " + new Date().toLocaleString();
}
setInterval(updateTime, 1000);//update time each second
updateTime();

/* ============================
   INFINITE BLOG SLIDER (3 POSTS)
   ============================ */
// get the elements from DOM
const track = document.querySelector(".slider-track");
const slides = Array.from(document.querySelectorAll(".blog-item"));
const prevBtn = document.querySelector(".prev");
const nextBtn = document.querySelector(".next");

let index = 1; // cus 0 is clone
let slideWidth;

// Clone first & last slides
const firstClone = slides[0].cloneNode(true);
const lastClone = slides[slides.length - 1].cloneNode(true);

firstClone.classList.add("clone");
lastClone.classList.add("clone");

track.appendChild(firstClone); // add first one ka clone at the end
track.insertBefore(lastClone, slides[0]); // add last one ka clone at the start
// idea is when index reaches the clone.. we replace it with real one so that seamlessly it looks like looped
const allSlides = Array.from(track.children);

function setSlideWidth() {
  slideWidth = allSlides[0].getBoundingClientRect().width + 20; // 20 = margin
  track.style.transform = `translateX(-${slideWidth * index}px)`;
}

window.addEventListener("resize", setSlideWidth);
setSlideWidth();

// Move slider
function moveToIndex() {
  track.style.transition = "transform 0.45s ease";
  track.style.transform = `translateX(-${slideWidth * index}px)`;
}

// Next / Prev buttons
nextBtn.onclick = () => {
  if (index >= allSlides.length - 1) return;
  index++;
  moveToIndex();
};

prevBtn.onclick = () => {
  if (index <= 0) return;
  index--;
  moveToIndex();
};

// Seamless loop reset
track.addEventListener("transitionend", () => {
  if (allSlides[index].classList.contains("clone")) {
    track.style.transition = "none";

    if (index === allSlides.length - 1) {
      index = 1;
    } else if (index === 0) {
      index = allSlides.length - 2;
    }

    track.style.transform = `translateX(-${slideWidth * index}px)`;
  }
});

// Auto-rotate forever
setInterval(() => {
  index++;
  moveToIndex();
}, 6000);

/* ============================
   CHAT BUTTON HANDLER
   ============================ */

const chatButton = document.getElementById("chatButton");
chatButton.addEventListener("click", () => {
  window.location.href = "/chat";
});

