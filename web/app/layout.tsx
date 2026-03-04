import type { Metadata } from "next";
import "./globals.css";
import { TopNav } from "@/components/layout/TopNav";
import { Providers } from "@/lib/query-client";
import { Toaster } from "sonner";

export const metadata: Metadata = {
  title: "台股 AI 預測系統",
  description: "AI 驅動的台股走勢預測 Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-TW">
      <body className="min-h-screen antialiased" style={{ background: "var(--bg-primary)", color: "var(--text-primary)" }}>
        {/* Film grain overlay */}
        <div className="grain-overlay" />
        {/* Ambient gold glow */}
        <div className="ambient-glow" />

        <Providers>
          <div className="relative z-10 flex flex-col min-h-screen">
            <TopNav />
            <main className="flex-1 overflow-auto">
              {children}
            </main>
          </div>
          <Toaster
            position="top-right"
            theme="dark"
            richColors
            toastOptions={{
              style: {
                background: "#141922",
                border: "1px solid rgba(255,255,255,0.08)",
                color: "#D8DBE2",
              },
            }}
          />
        </Providers>
      </body>
    </html>
  );
}
