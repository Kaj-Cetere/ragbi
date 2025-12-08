import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kesher AI - Torah Learning Assistant",
  description: "Ask questions about Halakha with AI-powered insights from Shulchan Arukh and classical commentaries",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
