import client from './client'

export interface UserInfo {
  id: number
  username: string
  email: string
  is_admin: boolean
  is_active: boolean
  created_at: string
  last_login: string | null
}

export interface TokenResponse {
  access_token: string
  token_type: string
  user: UserInfo
}

export const authApi = {
  register: (username: string, email: string, password: string) =>
    client.post<TokenResponse>('/auth/register', { username, email, password }),

  login: (username: string, password: string) =>
    client.post<TokenResponse>('/auth/login', { username, password }),

  me: () => client.get<UserInfo>('/auth/me'),

  listUsers: () => client.get<UserInfo[]>('/auth/admin/users'),

  updateUser: (id: number, data: { is_active?: boolean; is_admin?: boolean; password?: string }) =>
    client.patch<UserInfo>(`/auth/admin/users/${id}`, data),

  deleteUser: (id: number) => client.delete(`/auth/admin/users/${id}`),
}
