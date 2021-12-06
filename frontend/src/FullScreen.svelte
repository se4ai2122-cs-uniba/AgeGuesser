<script>
  import { onMount } from "svelte";
  export let data;
  export let show;

  let top, left;

  let faceId = "#face-" + data.id;
  let canvasId = "box_canvas_" + faceId.substr(1);
  let drawCanvas;
  let drawCtx;

  let canvasPadding = 20;


  function drawBoxes(clear, prediction) {

    if (drawCanvas == undefined) {
      drawCanvas = document.getElementById(canvasId);
      drawCtx = drawCanvas.getContext("2d");
    }

    // get top margin
    top = document.querySelector(faceId).getBoundingClientRect().top +
          document.querySelector(faceId).scrollTop

    // get left margin
    left = document.querySelector(faceId).getBoundingClientRect().left;

    if (document.querySelector(faceId) == undefined) return;

    // clean up canvas if needed
    if (clear) {
      drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    }

    // get image natural size
    let nw = data.width;
    let nh = data.height;

    // get actual size
    // since layout is responsive, it may change
    let currentw = document.querySelector(faceId).width;
    let currenth = document.querySelector(faceId).height;

    // update canvas accordingly
    drawCanvas.width = currentw + canvasPadding;
    drawCanvas.height = currenth + canvasPadding;

    drawCtx.lineWidth = 2;
    drawCtx.strokeStyle = "cyan";
    drawCtx.font = "14px Verdana";
    drawCtx.fillStyle = "cyan";

    for (let i = 0; i < prediction.length; i++) {
      let p = prediction[i];

      // we cannot draw the canvas yet, first we have to scale the coordinates
      // wrt the actual image size
      let percentBx = 100 * (p.x / nw), // x %
          percentBy = 100 * (p.y / nh), // y %
          percentBw = (p.w * 100) / nw, // width %
          percentBh = (p.h * 100) / nh; // height %

      // then map the values to the current canvas

      let finalBx = (percentBx * currentw) / 100, // x 
          finalBy = (percentBy * currenth) / 100, // y 
          finalBw = (percentBw * currentw) / 100, // width 
          finalBh = (percentBh * currenth) / 100; // height

      // set canvas props
      drawCtx.lineWidth = 2;
      drawCtx.font = "12px Verdana";
      drawCtx.strokeStyle = "white";

      // draw face bbox
      drawCtx.strokeRect(finalBx, finalBy + canvasPadding, finalBw, finalBh);

      // draw age label on top of it
      drawCtx.fillStyle = "white";
      drawCtx.fillRect(finalBx, finalBy + 5, finalBw, 15);
      drawCtx.fillStyle = "black";
      drawCtx.fillText(" " + p.age, finalBx, finalBy + canvasPadding - 2);

    }
  }

  onMount(() => {
    
    setTimeout(function () {
      drawBoxes(true, data.predictions);
    }, 800);
  });

</script>


<div id="imgfull">
  <button  on:click={() => (show = !show)}>Close</button>

  <img
    id={faceId.substr(1)}
    src={data.img}
    class="img-fluid mx-auto d-block"
    alt="your face"
    style=" max-height: 100%;"
  />

  <canvas
    id={canvasId}
    style="top: { top -
      canvasPadding}px; left: {left}px; position: absolute; padding: 0;"
  />


</div>

<style>
  #imgfull {
    background-color: white;
    position: fixed;
    top: 0;
    left: 0;
    z-index: 1;
    width: 100%;
    height: 100%;
    overflow: auto;
  }
</style>