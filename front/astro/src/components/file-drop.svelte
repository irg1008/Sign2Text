<script lang="ts">
	import { backIn, backOut, cubicOut } from "svelte/easing";
	import { fade, fly, scale } from "svelte/transition";
	import { getSignForVideo } from "services/sign.service";
	import Danger from "./danger-icon.svelte";

	let video: File;

	let dragCounter = 0;
	let isDragging = false;
	let error: string = "";
	let sign: string = "";
	let loadingSign = false;

	const cleanError = () => (error = "");

	const setDropEffect = (e: DragEvent) => {
		if (!e.dataTransfer) return;
		e.dataTransfer.dropEffect = "copy";
	};

	const setDragEffect = (e: DragEvent) => {
		if (!e.dataTransfer) return;
		e.dataTransfer.effectAllowed = "all";
	};

	const onDragIn = (_: DragEvent) => {
		dragCounter++;
		if (dragCounter !== 1) return;
		isDragging = true;
	};

	const onDragOut = (_: DragEvent) => {
		dragCounter--;
		if (dragCounter !== 0) return;
		isDragging = false;
	};

	const onDrop = (e: DragEvent) => {
		onDragOut(e);
		const file = e.dataTransfer?.files?.[0];

		if (file?.type !== "video/mp4") {
			error = "Only MP4 files are supported";
			return;
		}

		cleanError();
		sign = "";
		video = file;
	};

	const getSign = async () => {
		loadingSign = true;
		if (sign) return;
		const res = await getSignForVideo(video);
		sign = res.target;
		loadingSign = false;
	};
</script>

<svelte:window
	on:dragstart|preventDefault={setDragEffect}
	on:dragover|preventDefault={setDropEffect}
	on:dragenter|preventDefault={onDragIn}
	on:dragleave|preventDefault={onDragOut}
	on:drop|preventDefault={onDrop}
/>

{#if error}
	<span
		transition:fly={{ duration: 200, x: 100, easing: backOut }}
		on:click={cleanError}
		class="fixed bottom-0 right-0 flex items-center gap-4 p-6 m-8 font-bold uppercase bg-red-500 rounded-lg shadow-lg cursor-pointer text-neutral-50 shadow-red-800/10"
	>
		<Danger class="text-lg" />
		{error}
	</span>
{/if}

{#if video}
	{#key video.name}
		<article class="flex flex-col items-center gap-4 p-6 font-bold capitalize">
			<video
				autoplay
				loop
				muted
				width={600}
				class="rounded-lg shadow-xl shadow-neutral-600/20"
			>
				<source src={URL.createObjectURL(video)} type={video.type} />
				<track kind="captions" />
			</video>

			{#if sign}
				<h2
					in:fade={{ duration: 200, delay: 220 }}
					class="text-2xl font-bold text-neutral-800"
				>
					Sign is: <strong>{sign}</strong>
				</h2>
			{:else}
				<button
					out:fade={{ duration: 200 }}
					on:click={() => getSign()}
					disabled={loadingSign}
					class="px-4 py-2 text-xl font-bold uppercase transition-all duration-200 ease-in-out rounded-lg shadow-lg shadow-sky-300/60 active:shadow-md bg-sky-500 hover:bg-sky-600 active:bg-sky-800 text-neutral-50 disabled:pointer-events-none disabled:bg-opacity-60"
				>
					{loadingSign ? "Getting sign" : "Get sign"}
				</button>
			{/if}
		</article>
	{/key}
{/if}

{#if isDragging}
	<div
		transition:fade={{ duration: 300, easing: cubicOut }}
		class="fixed top-0 left-0 grid w-full h-full bg-neutral-200/50 place-content-center backdrop-blur-md"
	>
		<h1
			in:fly={{ duration: 200, easing: backOut, y: -100 }}
			out:scale={{ duration: 200, easing: backIn }}
			class="font-bold text-center uppercase text-8xl text-neutral-600"
		>
			Now drop it!
		</h1>
	</div>
{/if}
