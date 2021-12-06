<script>
  import { onMount } from "svelte";
  export let data;
  export let show;

  let top, left;

  let faceId = "#face-" + data.id;
  let canvasId = "box_canvas_" + faceId.substr(1);

  let canvasPadding = 20;

  onMount(() => {});
</script>

<div id="imgfull">
  <button on:click={() => (show = !show)}>Close</button>

  <img
    id={faceId.substr(1)}
    src={data.img}
    class="img-fluid mx-auto d-block"
    alt="your face"
    style=" max-height: 100%;"
  />

  <canvas
    id={canvasId}
    style="top: {top -
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
