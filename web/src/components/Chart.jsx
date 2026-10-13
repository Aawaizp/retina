import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, HeatmapChart, LineChart } from "echarts/charts";
import {
  GridComponent, LegendComponent, TooltipComponent, VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([BarChart, HeatmapChart, LineChart, GridComponent, LegendComponent, TooltipComponent,
  VisualMapComponent, CanvasRenderer]);

const dark = () => window.matchMedia?.("(prefers-color-scheme: dark)").matches;

export default function Chart({ option, height = 300 }) {
  const ref = useRef(null);
  const chart = useRef(null);

  useEffect(() => {
    chart.current = echarts.init(ref.current, dark() ? "dark" : null, { renderer: "canvas" });
    const resize = new ResizeObserver(() => chart.current?.resize());
    resize.observe(ref.current);
    return () => {
      resize.disconnect();
      chart.current?.dispose();
    };
  }, []);

  useEffect(() => {
    chart.current?.setOption({ backgroundColor: "transparent", ...option }, true);
  }, [option]);

  return <div ref={ref} style={{ width: "100%", height }} />;
}
