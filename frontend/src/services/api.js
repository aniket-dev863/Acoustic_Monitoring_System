import axios from 'axios';

const API_BASE = '/api';

export const fetchHealth = async () => {
  const res = await axios.get(`${API_BASE}/health`);
  return res.data;
};

export const fetchStats = async () => {
  const res = await axios.get(`${API_BASE}/stats`);
  return res.data;
};

export const fetchEvents = async (params = {}) => {
  const res = await axios.get(`${API_BASE}/events`, { params });
  return res.data;
};

export const fetchEventById = async (id) => {
  const res = await axios.get(`${API_BASE}/events/${id}`);
  return res.data;
};

export const createEvent = async (formData) => {
  const res = await axios.post(`${API_BASE}/events`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const fetchExperiments = async () => {
  const res = await axios.get(`${API_BASE}/experiments`);
  return res.data;
};
