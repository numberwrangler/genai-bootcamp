import { goto } from '$app/navigation';
import { checkIsAuthenticated } from '$lib/auth';

export const ssr = false;

export async function load({ url }) {
	const isAuthenticated = await checkIsAuthenticated();

	const onLoginPage = url.pathname === '/admin/login';

	if (onLoginPage) {
		if (isAuthenticated) {
			await goto('/admin/');
		}
	} else {
		if (!isAuthenticated) {
			await goto('/admin/login');
		}
	}

	return {
		isAuthenticated
	};
}
