<script>
    import FullScreen from "./FullScreen.svelte";

    let show = false;
    export let data;

    async function showImg() {
        data.height = document.querySelector("#f-" + data.id).naturalHeight;
        data.width = document.querySelector("#f-" + data.id).naturalWidth;

        show = !show;
    }
</script>

<div class="card mb-2" style="width: 18rem;  ">
    <img
        src={data.img}
        id="f-{data.id}"
        style="max-height: 150px; object-fit: cover;"
        alt="..."
    />
    <div class="card-body">
        {#if data.isLoading}
            <div class="sk-chase">
                <div class="sk-chase-dot" />
                <div class="sk-chase-dot" />
                <div class="sk-chase-dot" />
                <div class="sk-chase-dot" />
                <div class="sk-chase-dot" />
                <div class="sk-chase-dot" />
            </div>
        {:else if data.predictions.length > 0}
            <p>
                {#if data.extract_faces}
                {data.predictions.length} face{#if data.predictions.length > 1}s{/if}
                found. 
                {/if}
                
                {#if data.predictions.length == 1}
                Age: {data.predictions[0].age}
                {/if}
            </p>

            {#if data.predictions.length > 0}
            <button class="btn btn-primary" on:click={showImg}>Show</button>
            {/if}
        {/if}
    </div>
</div>

{#if show}
    <FullScreen bind:data bind:show />
{/if}
