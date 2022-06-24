const endpoint = import.meta.env.PUBLIC_SIGN_URI;

interface Sign {
	target: string;
}

export const getSignForVideo = async (video: File): Promise<Sign> => {
	// Add file to multiform then send a get request to the sign endpoint.

	const formData = new FormData();
	formData.append("video", video);

	const res = await fetch(`${endpoint}/sign`, {
		method: "POST",
		body: formData,
	});

	return await res.json();
};
