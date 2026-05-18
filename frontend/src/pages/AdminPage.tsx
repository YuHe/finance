import { useState, useEffect } from 'react'
import { authApi } from '../api/auth'
import type { UserInfo } from '../api/auth'
import { useAuth } from '../store/authStore'

function Badge({ active }: { active: boolean }) {
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${
      active ? 'bg-green-900 text-green-400' : 'bg-gray-700 text-gray-400'
    }`}>
      {active ? '正常' : '禁用'}
    </span>
  )
}

export default function AdminPage() {
  const { user: currentUser } = useAuth()
  const [users, setUsers] = useState<UserInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [editId, setEditId] = useState<number | null>(null)
  const [newPassword, setNewPassword] = useState('')
  const [saving, setSaving] = useState(false)

  const load = async () => {
    try {
      const res = await authApi.listUsers()
      setUsers(res.data)
    } catch {
      setError('加载用户列表失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const toggleActive = async (u: UserInfo) => {
    setSaving(true)
    try {
      await authApi.updateUser(u.id, { is_active: !u.is_active })
      setUsers((prev) => prev.map((x) => x.id === u.id ? { ...x, is_active: !u.is_active } : x))
    } catch (err: any) {
      setError(err.response?.data?.detail || '操作失败')
    } finally {
      setSaving(false)
    }
  }

  const toggleAdmin = async (u: UserInfo) => {
    setSaving(true)
    try {
      await authApi.updateUser(u.id, { is_admin: !u.is_admin })
      setUsers((prev) => prev.map((x) => x.id === u.id ? { ...x, is_admin: !u.is_admin } : x))
    } catch (err: any) {
      setError(err.response?.data?.detail || '操作失败')
    } finally {
      setSaving(false)
    }
  }

  const resetPassword = async (u: UserInfo) => {
    if (!newPassword.trim()) return
    setSaving(true)
    try {
      await authApi.updateUser(u.id, { password: newPassword })
      setNewPassword('')
      setEditId(null)
    } catch (err: any) {
      setError(err.response?.data?.detail || '重置失败')
    } finally {
      setSaving(false)
    }
  }

  const deleteUser = async (u: UserInfo) => {
    if (!confirm(`确定删除用户 "${u.username}"？此操作不可撤销。`)) return
    setSaving(true)
    try {
      await authApi.deleteUser(u.id)
      setUsers((prev) => prev.filter((x) => x.id !== u.id))
    } catch (err: any) {
      setError(err.response?.data?.detail || '删除失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl font-bold text-white">用户管理</h1>
        <p className="text-gray-400 text-sm mt-1">管理系统用户账号、权限和状态</p>
      </div>

      {error && (
        <div className="mb-4 bg-red-900/30 border border-red-800 rounded-lg px-4 py-3 text-red-400 text-sm flex justify-between">
          {error}
          <button onClick={() => setError('')} className="text-red-300 hover:text-white">✕</button>
        </div>
      )}

      {loading ? (
        <div className="text-gray-400 text-sm">加载中...</div>
      ) : (
        <div className="bg-gray-800 rounded-xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-gray-400 text-xs">
                <th className="text-left px-4 py-3">用户名</th>
                <th className="text-left px-4 py-3">邮箱</th>
                <th className="text-left px-4 py-3">状态</th>
                <th className="text-left px-4 py-3">权限</th>
                <th className="text-left px-4 py-3">最后登录</th>
                <th className="text-left px-4 py-3">注册时间</th>
                <th className="text-left px-4 py-3">操作</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                  <td className="px-4 py-3 text-white font-medium">
                    {u.username}
                    {u.id === currentUser?.id && (
                      <span className="ml-2 text-xs text-blue-400">(我)</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-gray-300">{u.email}</td>
                  <td className="px-4 py-3"><Badge active={u.is_active} /></td>
                  <td className="px-4 py-3">
                    <span className={`text-xs ${u.is_admin ? 'text-yellow-400' : 'text-gray-400'}`}>
                      {u.is_admin ? '管理员' : '普通用户'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-xs">
                    {u.last_login ? new Date(u.last_login).toLocaleDateString('zh-CN') : '—'}
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-xs">
                    {new Date(u.created_at).toLocaleDateString('zh-CN')}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2 flex-wrap">
                      {/* 禁用/启用 */}
                      {u.id !== currentUser?.id && (
                        <button
                          onClick={() => toggleActive(u)}
                          disabled={saving}
                          className={`text-xs px-2 py-1 rounded ${
                            u.is_active
                              ? 'bg-red-900/40 text-red-400 hover:bg-red-900/70'
                              : 'bg-green-900/40 text-green-400 hover:bg-green-900/70'
                          } disabled:opacity-50`}
                        >
                          {u.is_active ? '禁用' : '启用'}
                        </button>
                      )}
                      {/* 设置管理员 */}
                      {u.id !== currentUser?.id && (
                        <button
                          onClick={() => toggleAdmin(u)}
                          disabled={saving}
                          className="text-xs px-2 py-1 rounded bg-yellow-900/40 text-yellow-400 hover:bg-yellow-900/70 disabled:opacity-50"
                        >
                          {u.is_admin ? '撤销管理员' : '设为管理员'}
                        </button>
                      )}
                      {/* 重置密码 */}
                      {editId === u.id ? (
                        <div className="flex items-center gap-1">
                          <input
                            type="password"
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            placeholder="新密码"
                            className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs w-24 focus:outline-none focus:border-blue-500"
                          />
                          <button
                            onClick={() => resetPassword(u)}
                            disabled={saving || !newPassword}
                            className="text-xs px-2 py-1 rounded bg-blue-700 text-white hover:bg-blue-600 disabled:opacity-50"
                          >
                            确认
                          </button>
                          <button
                            onClick={() => { setEditId(null); setNewPassword('') }}
                            className="text-xs px-2 py-1 rounded bg-gray-600 text-gray-300 hover:bg-gray-500"
                          >
                            取消
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setEditId(u.id)}
                          className="text-xs px-2 py-1 rounded bg-gray-600 text-gray-300 hover:bg-gray-500"
                        >
                          重置密码
                        </button>
                      )}
                      {/* 删除 */}
                      {u.id !== currentUser?.id && (
                        <button
                          onClick={() => deleteUser(u)}
                          disabled={saving}
                          className="text-xs px-2 py-1 rounded bg-red-900/40 text-red-400 hover:bg-red-900/70 disabled:opacity-50"
                        >
                          删除
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {users.length === 0 && (
            <div className="text-center py-12 text-gray-500">暂无用户</div>
          )}
        </div>
      )}
    </div>
  )
}
