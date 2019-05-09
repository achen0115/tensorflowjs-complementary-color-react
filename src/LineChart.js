import React from 'react';

import { LineChart, XAxis, YAxis, CartesianGrid, Line } from 'recharts';

export default function LnChart({ data, dataKey }) {
  return (
    <LineChart width={300} height={300} data={data}>
      <XAxis dataKey="name" />
      <YAxis />
      <CartesianGrid stroke="#eee" strokeDasharray="5 5" />
      <Line type="monotone" dataKey={dataKey} stroke="#8884d8" />
    </LineChart>
  );
}
