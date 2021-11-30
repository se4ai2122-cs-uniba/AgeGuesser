<script>
    export let data;
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
        {:else }         
        
        {#if data.predictions.length > 0}

            <p>
                
                {data.predictions.length} face{#if data.predictions.length > 1}s{/if}
                found.
            </p>
        {/if}    

        {/if}
    </div>
</div>

<style>
</style>
