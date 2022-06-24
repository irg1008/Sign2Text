<script lang="ts">
	import { fade, fly, scale } from "svelte/transition";
	import { backIn, backOut, cubicOut } from "svelte/easing";

	const videoType = "video/mp4";

	let videoUrl: string;
	let dragCounter = 0;
	let isDragging = false;
	let error: string = "";

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

		if (file?.type !== videoType) {
			error = "Only MP4 files are supported";
			return;
		}

		error = "";
		videoUrl = URL.createObjectURL(file);
	};
</script>

<svelte:window
	on:dragstart|preventDefault={setDragEffect}
	on:dragover|preventDefault={setDropEffect}
	on:dragenter|preventDefault={onDragIn}
	on:dragleave|preventDefault={onDragOut}
	on:drop|preventDefault={onDrop}
/>

<h1 class="capitalize text-4xl text-neutral-400 text-center leading-relaxed">
	Drag and drop your video here
</h1>

{#if error}
	<span>{error}</span>
{/if}

{#if videoUrl}
	{#key videoUrl}
		<video autoplay controls>
			<source src={videoUrl} type={videoType} />
			<track kind="captions" />
		</video>
	{/key}
{/if}

{#if isDragging}
	<div
		transition:fade={{ duration: 300, easing: cubicOut }}
		class="fixed h-full w-full top-0 left-0 bg-neutral-200/50 grid place-content-center backdrop-blur-md"
	>
		<h1
			in:fly={{ duration: 200, easing: backOut, y: -100 }}
			out:scale={{ duration: 200, easing: backIn }}
			class="uppercase font-bold text-center text-8xl text-neutral-600"
		>
			Now drop it!
		</h1>
	</div>
{/if}
