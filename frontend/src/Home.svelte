<script>

	import { getModels, upload } from "./api.js"

	let fileinput;

	// image id
	let i = 0;

	// submitted images
  let imgs = [];

	// queue of images to be uploaded
  let imgsQueue = [];

	// Variable to be listened for, in order to trigger the upload. 
	// It will be changed when all the images selected by the user have been successfully 
	// uploaded from the client.
	let imgsToBeHandled = undefined;

	// keep track of the number of images uploaded.
	// When toElaborate == #selected images => can trigger ageguesser backend endpoint.
  let toElaborate = 0;

	let ready = 2
	/*
		Handle image(s) upload.
	*/
	
  const onFileSelected = async (e) => {

    imgsQueue = [];
		imgsToBeHandled = undefined
    toElaborate = 0;
		ready = 2;
    for (let index = 0; index < e.target.files.length; index++) {
      const g = e.target.files[index];

      let newImg = {
        id: i++,
        img: URL.createObjectURL(g),
        predictions: []
				
      };
      imgs.push(newImg);

      let reader = new FileReader();
      reader.readAsBinaryString(g);

      reader.onload = async (e) => {

				console.log("LOADED")
        if (e.total == e.loaded) {
          imgsQueue.push({
            fileInput: fileinput.files[index],
            img: newImg
          });
          toElaborate++;
        }

				if (toElaborate == fileinput.files.length){

					ready = 0
					console.log("done")

				}

        
      };
    }
  };

	async function go(){

    imgsToBeHandled = imgsQueue;
    
	}

  async function handleQueue(queue) {
    if (queue == undefined) return;

		ready = 1

    for (let index = 0; index < queue.length; index++) {
      const img = queue[index];

			console.log("SENDING... " + img.img.id);

			/* todo: get values from form */
			let payload = {
				"file" : img.fileInput,
				"model" : "EfficientNet B0",
				"extract_faces": true 
			}

			let res = await upload(payload,);

			img.img.predictions = res.faces;

			console.log(img.img.predictions)
			console.log("DONE " + img.img.id + "");

    }
		ready = 2

	}

	// register listener on imgsToBeHandled
	// It will start uploading each image to the backend 
	// as soon as each of them are loaded
	$: handleQueue(imgsToBeHandled);

</script>

<div class="container">

	<div class="row ">
		<div class="col-12 text-center">
			<h1>Age Guesser v0.1</h1>

		</div>
	</div>

	<div class="row justify-content-center">

	
	<div class="col-sm-6" style="border: solid #dee2e6; border-width: 1px; border-radius: 0.25rem; padding: 1.5rem;">
		<input
		type="file"
		multiple="multiple"
		style="display: none;"
		accept="*.jpg *.png *.jpeg"
		on:change={async (e) => await onFileSelected(e)}
		bind:this={fileinput}
	/>


		<div class="row mb-3">
			<label for="imgFileUpload" class="col-sm-4 col-form-label">Choose image(s)</label>
			<div class="col-sm-8">
				<button class="btn btn-primary" id="imgFileUpload" on:click={() => fileinput.click()}>Upload</button>
			</div>
		</div>

		<div class="row mb-3">
			<label for="inputPassword3" class="col-sm-4 col-form-label">Extract faces</label>
			<div class="col-sm-8">
				<input class="form-check-input" type="checkbox" id="inputPassword3" checked>
			</div>
		</div>
		<fieldset class="row mb-3">
			<legend class="col-form-label col-sm-4 pt-0">Model</legend>
			<div class="col-sm-8">
				<!-- todo: get available models from backend -->
				<select class="form-select" aria-label="Model select">
					<option value="EfficientNet B0" selected>EfficientNet B0</option>
					<option value="EfficientNetv2 B0">EfficientNetv2 B0</option>
				</select>
				
			</div>
		</fieldset>
		{#if ready == 0}
			<button class="btn btn-primary" on:click={() => go()}>Go</button>
		{:else if ready == 1}
			 <!-- else if content here -->
			<button class="btn btn-primary" disabled>Processing..</button>
		{:else if ready == 2}
			<!-- else if content here -->
		 <button class="btn btn-primary" style="display: none;">Go</button>
		{/if}
		
	</div>
</div>
</div>
	

<style>
	main {
		text-align: center;
		padding: 1em;
		
		margin: 0 auto;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

</style>