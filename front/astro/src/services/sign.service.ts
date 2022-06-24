const ENDPOINT = "http://localhost:8000";

interface Sign {
	target: string;
}

export const getSignForVideo = async (video: File): Promise<Sign> => {
	// Add file to multiform then send a get request to the sign endpoint.

	const formData = new FormData();
	formData.append("video", video);

	const res = await fetch(`${ENDPOINT}/sign`, {
		method: "POST",
		body: formData,
	});

	return await res.json();
};
