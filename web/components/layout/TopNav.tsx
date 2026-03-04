"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { BarChart3, Clock } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", icon: BarChart3, label: "分析" },
  { href: "/history", icon: Clock, label: "紀錄" },
];

export function TopNav() {
  const pathname = usePathname();

  return (
    <header
      className="sticky top-0 z-50 flex h-14 items-center justify-between px-6 shrink-0"
      style={{
        background: "rgba(11,14,22,0.85)",
        backdropFilter: "blur(16px) saturate(180%)",
        borderBottom: "1px solid var(--border)",
      }}
    >
      {/* Left: Logo */}
      <Link href="/" className="flex items-center gap-2.5 shrink-0">
        <div className="relative flex h-7 w-7 items-center justify-center">
          <div
            className="absolute inset-0 rounded-lg"
            style={{
              background: "linear-gradient(135deg, rgba(201,168,76,0.15), rgba(201,168,76,0.05))",
              border: "1px solid rgba(201,168,76,0.2)",
            }}
          />
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="relative">
            <path d="M2 12L5 4L8 9L11 2L14 8" stroke="#C9A84C" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            <circle cx="14" cy="8" r="1.5" fill="#C9A84C"/>
          </svg>
        </div>
        <span className="font-display text-sm tracking-wider" style={{ color: "var(--accent-gold)" }}>
          AURUM
        </span>
      </Link>

      {/* Center: Navigation */}
      <nav className="flex items-center gap-1">
        {NAV_ITEMS.map((item) => {
          const active = item.href === "/"
            ? pathname === "/"
            : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "relative flex items-center gap-2 px-4 py-1.5 rounded-lg text-[13px] font-medium transition-all duration-200",
                active
                  ? "text-[#C9A84C]"
                  : "text-[#4B5263] hover:text-[#8B90A0]"
              )}
            >
              <item.icon className="h-[15px] w-[15px]" />
              <span>{item.label}</span>
              {active && (
                <div
                  className="absolute bottom-0 left-3 right-3 h-[2px] rounded-full"
                  style={{ background: "var(--accent-gold)" }}
                />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Right: Version badge */}
      <div className="flex items-center gap-3 shrink-0">
        <span
          className="text-[8px] tracking-[0.2em] uppercase"
          style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
        >
          v2.0
        </span>
      </div>
    </header>
  );
}
