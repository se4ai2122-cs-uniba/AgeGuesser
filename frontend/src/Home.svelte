<script>
	import { getModels, upload } from "./api.js";
	import { onMount } from 'svelte';
	import Card from "./Card.svelte";
	import exifr from 'exifr'

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

	let ready = 2;

	let extract_faces = true

	let selected_model = {}
	let models_meta = []

	onMount(async () => {
		
		let _models_meta = (await getModels()).data
		Object.keys(_models_meta).forEach(key => {
		
			models_meta.push({ "key": key,"name": _models_meta[key].backbone.replace("-", " ") })
		});

		selected_model = models_meta[0]

		models_meta = models_meta
	});

	/*
		Handle image(s) upload.
	*/

	const onFileSelected = async (e) => {
		imgsToBeHandled = undefined;
		toElaborate = 0;
		ready = 2;
		for (let index = 0; index < e.target.files.length; index++) {
			const g = e.target.files[index];

			let newImg = {
				id: i++,
				img: URL.createObjectURL(g),
				predictions: [],
				isLoading: false,
			};

			imgs.push(newImg);

			imgs = imgs;

			let reader = new FileReader();
			reader.readAsBinaryString(g);

			reader.onload = async (e) => {
				console.log("LOADED");
				if (e.total == e.loaded) {
					imgsQueue.push({
						fileInput: fileinput.files[index],
						img: newImg,
					});
					toElaborate++;
				}

				if (toElaborate == fileinput.files.length) {
					ready = 0;
					console.log("done");
				}
			};
		}
	};

	async function go() {
		/* console.log(selected_model)
		console.log(extract_faces) */
		imgsToBeHandled = imgsQueue;
	}

	async function handleQueue(queue) {
		if (queue == undefined) return;

		ready = 1;

		for (let index = 0; index < queue.length; index++) {
			const img = queue[index];
			img.img.isLoading = true;
			imgs = imgs

			// mobile cameras often rotate images, even when they appear to be displayed in portrait
			// get orientation metadata from image { 3: 180°, 6: 270°, 8: 90° }
			let orientation = await exifr.orientation(document.getElementById("f-" + img.img.id)) 

			//console.log(orientation)

			console.log("SENDING... " + img.img.id);

			let payload = {
				file: img.fileInput,
				model: selected_model.key,
				extract_faces: extract_faces,
				orientation: orientation == undefined ? 0 : orientation
			};

			let res = await upload(payload);

			img.img.predictions = res.faces;
			img.img.isLoading = false;

			imgs = imgs;

			console.log(img.img.predictions);
			console.log("DONE " + img.img.id + "");
		}
		ready = 2;
		imgsQueue = [];
		let input = document.querySelector('input[type="file"]');
		input.value= ""

	}

	// register listener on imgsToBeHandled
	// It will start uploading each image to the backend
	// as soon as each of them are loaded
	$: handleQueue(imgsToBeHandled);
</script>

<div class="container">
	<div class="row ">
		<div class="col-12 text-center">
			<h1>
				<img src="/magic-wand.png" alt="icon" style="width: 3rem;" /> Age
				Guesser v0.1
			</h1>
		</div>
	</div>

	<div class="row justify-content-center">
		<div
			class="col-sm-6"
			style="border: solid #dee2e6; border-width: 1px; border-radius: 0.25rem; padding: 1.5rem;"
		>
			<input
				type="file"
				multiple="multiple"
				style="display: none;"
				accept="*.jpg *.png *.jpeg"
				on:change={async (e) => await onFileSelected(e)}
				bind:this={fileinput}
			/>

			<div class="row mb-3">
				<label for="imgFileUpload" class="col-sm-4 col-form-label"
					>Choose image(s)</label
				>
				<div class="col-sm-8">
					<button
						class="btn btn-primary"
						id="imgFileUpload"
						on:click={() => fileinput.click()}>Upload</button
					>
				</div>
			</div>

			<div class="row mb-3">
				<label for="inputPassword3" class="col-sm-4 col-form-label"
					>Extract faces</label
				>
				<div class="col-sm-8">
					<input
						class="form-check-input"
						type="checkbox"
						id="inputPassword3"
						bind:checked={extract_faces}						
					/>
				</div>
			</div>
			<fieldset class="row mb-3">
				<legend class="col-form-label col-sm-4 pt-0">Model</legend>
				<div class="col-sm-8">

					<select class="form-select" aria-label="Model select" bind:value={selected_model}>

						{#each models_meta as item}
						<option value={item}>{item.name}</option>
						{/each}
					</select>
				</div>
			</fieldset>
			{#if ready == 0}
				<button class="btn btn-primary" on:click={() => go()}>Go</button
				>
			{:else if ready == 1}
				<!-- else if content here -->
				<button class="btn btn-primary" disabled>Processing..</button>
			{:else if ready == 2}
				<!-- else if content here -->
				<button class="btn btn-primary" style="display: none;"
					>Go</button
				>
			{/if}
		</div>
	</div>

	<div class="row mt-3">
		{#each imgs as img}
			<div class="col-sm-12 col-md-6 col-xl-3">
				<Card bind:data={img} />
			</div>
		{/each}
	</div>
</div>

<style>
	
	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}
</style>
