import React from "react";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Theme } from "@radix-ui/themes";
import "@radix-ui/themes/styles.css";
import { App } from "./_components/App";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "LLM hallucination detection",
  description: "LLM hallucination detection app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Theme>
          <App>{children}</App>
        </Theme>
      </body>
    </html>
  );
}
