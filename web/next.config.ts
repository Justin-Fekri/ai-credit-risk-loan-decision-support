import type { NextConfig } from "next";

const API_URL = process.env.API_URL || "http://127.0.0.1:45232";

const nextConfig: NextConfig = {
  agentRules: false,
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${API_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
