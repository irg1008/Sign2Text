<script lang="ts">
	import { fade, fly, scale } from "svelte/transition";
	import { backIn, backOut, cubicOut } from "svelte/easing";

	let dragCounter = 0;
	let isDragging = false;

	const setDropEffect = (e: DragEvent) => {
		if (!e.dataTransfer) return;
		e.dataTransfer.dropEffect = "copy";
	};

	const setDragEffect = (e: DragEvent) => {
		if (!e.dataTransfer) return;
		e.dataTransfer.effectAllowed = "copy";
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
		if (file?.type !== "video/mp4") return;
		console.log("is mp4");
	};
</script>

<svelte:window
	on:dragstart|preventDefault={setDragEffect}
	on:dragover|preventDefault={setDropEffect}
	on:dragenter|preventDefault={onDragIn}
	on:dragleave|preventDefault={onDragOut}
	on:drop|preventDefault={onDrop}
/>

{#if isDragging}
	<div
		transition:fade={{ duration: 300, easing: cubicOut }}
		class="fixed h-full w-full top-0 left-0 bg-neutral-50/50 grid place-content-center backdrop-blur-md"
	>
		<h1
			in:fly={{ duration: 200, easing: backOut, y: -100 }}
			out:scale={{ duration: 200, easing: backIn }}
			class="uppercase font-bold text-center text-8xl"
		>
			Now drop it!
		</h1>
	</div>
{/if}
