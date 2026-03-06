"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { BarChart3, Clock } from "lucide-react";

const NAV_ITEMS = [
  { href: "/", icon: BarChart3, label: "分析" },
  { href: "/history", icon: Clock, label: "紀錄" },
];

export function TopNav() {
  const pathname = usePathname();

  const { data: overview } = useQuery({
    queryKey: ["market-overview-nav"],
    queryFn: () => api.getMarketOverview(),
    staleTime: 10 * 60 * 1000,
  });

  const scanDate = overview?.scan_date;
  const stockCount = overview?.stocks?.length ?? 0;

  return (
    <header
      className="sticky top-0 z-50 flex h-9 items-center justify-between px-4 shrink-0"
      style={{
        background: "rgba(11,14,22,0.9)",
        backdropFilter: "blur(16px) saturate(180%)",
        borderBottom: "1px solid var(--border)",
      }}
    >
      {/* Left: Logo */}
      <Link href="/" className="flex items-center gap-1.5 shrink-0">
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <path d="M2 12L5 4L8 9L11 2L14 8" stroke="#C9A84C" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          <circle cx="14" cy="8" r="1.5" fill="#C9A84C"/>
        </svg>
        <span className="font-display text-xs tracking-wider" style={{ color: "var(--accent-gold)" }}>
          台股預測
        </span>
      </Link>

      {/* Center: Navigation */}
      <nav className="flex items-center gap-0.5">
        {NAV_ITEMS.map((item) => {
          const active = item.href === "/"
            ? pathname === "/"
            : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "relative flex items-center gap-1.5 px-3 py-1 rounded text-[11px] font-medium transition-all duration-200",
                active
                  ? "text-[#C9A84C]"
                  : "text-[#4B5263] hover:text-[#8B90A0]"
              )}
            >
              <item.icon className="h-3 w-3" />
              <span>{item.label}</span>
              {active && (
                <div
                  className="absolute bottom-0 left-2 right-2 h-[1.5px] rounded-full"
                  style={{ background: "var(--accent-gold)" }}
                />
              )}
            </Link>
          );
        })}
      </nav>

      {/* Right: Market status + Version */}
      <div className="flex items-center gap-2 shrink-0">
        {scanDate && (
          <span
            className="rounded px-1.5 py-px text-[8px] font-medium"
            style={{
              background: "rgba(201,168,76,0.06)",
              color: "var(--text-secondary)",
              border: "1px solid rgba(201,168,76,0.1)",
            }}
          >
            {scanDate} · {stockCount}檔
          </span>
        )}
        <span
          className="text-[7px] tracking-[0.15em] uppercase"
          style={{ color: "var(--text-muted)", fontFamily: "'Space Mono', monospace" }}
        >
          v3.0
        </span>
      </div>
    </header>
  );
}
