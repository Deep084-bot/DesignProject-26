import React, { useState, useEffect } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { Plus, Server, PlayCircle } from 'lucide-react';
import { getExecutionMode, setExecutionMode, getBackendBaseUrl } from '../../services/api';
import type { ExecutionMode } from '../../types/inspection';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const [mode, setMode] = useState<ExecutionMode>(getExecutionMode());
  const backendBaseUrl = getBackendBaseUrl();

  useEffect(() => {
    const handleModeChange = (e: Event) => {
      const customEvent = e as CustomEvent<ExecutionMode>;
      if (customEvent.detail) {
        setMode(customEvent.detail);
      } else {
        setMode(getExecutionMode());
      }
    };
    window.addEventListener('surface-mode-changed', handleModeChange);
    return () => window.removeEventListener('surface-mode-changed', handleModeChange);
  }, []);

  const toggleMode = () => {
    const nextMode: ExecutionMode = mode === 'backend' ? 'demo' : 'backend';
    setExecutionMode(nextMode);
    setMode(nextMode);
  };

  const navLinks = [
    { to: '/', label: 'Overview' },
    { to: '/new', label: 'New inspection' },
    { to: '/inspections', label: 'Inspections' },
    { to: '/nodes', label: 'Nodes' },
  ];

  return (
    <header className="sticky top-0 z-40 bg-[#F8F7F2]/90 backdrop-blur-md border-b border-[#E8E5DC]">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-7 h-7 rounded-md bg-[#9BAE98]/20 border border-[#9BAE98]/50 flex items-center justify-center text-[#4A5B48] transition-colors group-hover:bg-[#9BAE98]/30">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="12 2 2 7 12 12 22 7 12 2" />
                <polyline points="2 17 12 22 22 17" />
                <polyline points="2 12 12 17 22 12" />
              </svg>
            </div>
            <span className="font-semibold text-base text-[#343832] tracking-tight">
              Surface
            </span>
          </Link>

          {/* Navigation Items */}
          <nav className="flex items-center gap-1 sm:gap-2">
            {navLinks.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-[#E7EDE4] text-[#343832] font-semibold'
                      : 'text-[#777D73] hover:text-[#343832] hover:bg-[#F0EFE8]'
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>

        {/* Right Status & Action */}
        <div className="flex items-center gap-3">
          {/* Explicit Mode Indicator & Switcher */}
          <button
            type="button"
            onClick={toggleMode}
            title={
              mode === 'backend'
                ? `Active: Real processing mode (${backendBaseUrl || 'URL unset'}). Click to switch to Demo mode.`
                : 'Active: Demonstration mode (local simulation). Click to switch to Real Backend mode.'
            }
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium border transition-colors cursor-pointer ${
              mode === 'backend'
                ? 'bg-[#E7EDE4] text-[#3F513D] border-[#D5E0D1] hover:bg-[#DCE5D8]'
                : 'bg-[#F0EFE8] text-[#777D73] border-[#E8E5DC] hover:text-[#343832] hover:bg-[#E8E5DC]'
            }`}
          >
            {mode === 'backend' ? (
              <>
                <Server className="w-3 h-3 text-[#3F513D]" />
                <span>Real backend</span>
              </>
            ) : (
              <>
                <PlayCircle className="w-3 h-3 text-[#9BA297]" />
                <span>Demo mode</span>
              </>
            )}
          </button>

          {location.pathname !== '/new' && (
            <Link
              to="/new"
              className="inline-flex items-center gap-1.5 text-xs font-medium px-3.5 py-1.5 rounded-lg bg-[#9BAE98] hover:bg-[#8A9E87] active:bg-[#7D917A] text-white shadow-xs border border-[#8A9E87] transition-all"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New inspection</span>
            </Link>
          )}
        </div>
      </div>
    </header>
  );
};
